from typing import Union, Optional, Literal, Sequence, List, Dict, Any, Callable

import os
import tempfile
from pympi import Elan
from tqdm import tqdm
from transformers import pipeline, Pipeline

from zugubul.vad_to_elan import label_speech_segments
from zugubul.models.dataset import process_annotation_length
from zugubul.elan_tools import snip_audio, trim
from zugubul.main import init_annotate_parser, handle_annotate
import argparse
import pandas as pd

# enable pandas progress bars
tqdm.pandas()

def pipe_out_to_sli_label(pipe_out: List[Dict[str, str]]) -> Dict[str, str]:
    max_score = 0
    pred = ''
    for score_dict in pipe_out:
        score, label = score_dict['score'], score_dict['label']
        if score > max_score:
            pred = label
            max_score = score
    return {'sli_label': pred}

def pipe_out_to_sli_probs(pipe_out: List[Dict[str, str]]) -> Dict[str, str]:
    scores = {}
    for score_dict in pipe_out:
        score, label = score_dict['score'], score_dict['label']
        scores[label] = score
    return {'sli_scores': scores}

def infer(
    input_file: Union[str, List[str], pd.Series],
    model_path: Union[str, os.pathlike],
    task: Literal['asr', 'sli'] = 'asr',
    sli_out_format: Literal['labels', 'probs'] = 'labels',
    do_vad: bool = True,      
):
    # define relevant functions for ASR and SLI
    if task == 'asr':
        out_format_funct = lambda x:x
        pipe_class = 'automatic-speech-recognition'
    elif sli_out_format == 'labels':
        out_format_funct = pipe_out_to_sli_label
        pipe_class = 'audio-classification'
    else:
        out_format_funct = pipe_out_to_sli_probs
        pipe_class = 'audio-classification'
    pipe = pipeline(pipe_class, model_path)

    # single file w/ VAD
    if type(input_file) is str and do_vad:
        print('Performing VAD on file...')
        vad_eaf = label_speech_segments(input_file)
        sli_segs = _infer_on_segs(input_file, vad_eaf, out_format_funct, pipe)
        return sli_segs
    # multiple files w/ VAD
    elif do_vad:
        print('Performing VAD on files...')
        sli_outs = []
        for file in tqdm(input_file):
            sli_outs.extend(_infer_on_segs(file, out_format_funct, pipe))
        return sli_out
    # single file no VAD
    elif type(input_file) is str:
        print('Performing SLI on file...')
        sli_out = out_format_funct(pipe(input_file))
        return {'filename': input_file, **sli_out}
    # multiple files no VAD
    print('Performing SLI on files...')
    source_file = pd.Series(source_file) if type(source_file) is list else source_file
    sli_outs = _do_infer_list(input_file, pipe, out_format_funct)
    return [{'filename': file, **sli_out} for file, sli_out in zip(input_file, sli_outs)]

def _infer_on_segs(
    input_file: Union[str, os.PathLike],
    segs: Union[str, os.PathLike, Elan.Eaf, pd.DataFrame],
    pipe: Pipeline,
    out_format_funct: Callable,
) -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        vad_df = snip_audio(annotations=segs, out_dir=tmpdir, audio=input_file)
        vad_segs = vad_df['wav_clip']

        tqdm.write('Performing SLI on VAD segments...')
        pipe_out = _do_infer_list(vad_segs, pipe, out_format_funct)

    seg_outs = [{
            'file': input_file,
            'segments': [
                {'start': start, 'end': end, **seg_out}
                for start, end, seg_out in zip(vad_df['start'], vad_df['end'], pipe_out)
            ]
        }]
    
    return seg_outs

def _do_infer_list(
    input_list: pd.Series,
    pipe: Pipeline,
    out_format_funct: Callable,
):
    out_list = input_list.progress_apply(pipe)
    return out_list.apply(out_format_funct)

def annotate(
        input_file: Union[str, os.PathLike, List[str]],
        asr_model: Union[str, os.PathLike, None] = None,
        tgt_lang: Union[str, List[str], None] = None,
        lang_to_asr: Optional[Dict[str, str]] = None,
        sli_model: Union[str, os.PathLike, None] = None,
        sli_out_format: Literal['labels', 'probs'] = 'labels',
        do_vad: bool = True,
        etf: Optional[Union[str, Elan.Eaf]] = None,
    ) -> Elan.Eaf:
    """
    Perform SLI on source file using model at sli_model (a filepath or URL).
    Remove all annotations not belonging to the target language, then run ASR
    using model at asr_model.
    If tier is provided, add annotations to tier of that name.
    If etf is provided, use as template for output .eaf file.
    """
    
    if (tgt_lang or lang_to_asr) and (not sli_model):
        raise ValueError("Need to pass `sli_model` if passing either `tgt_lang` or `lang_to_asr`")

    # handle single and multi-file annotation identically
    if type(input_file) is str:
        input_file == [input_file,]

    # treat single-language ASR and multi-language ASR identically
    if (asr_model and tgt_lang) and (not lang_to_asr):
        lang_to_asr = {tgt_lang: asr_model}

    lang_specific_asr = tgt_lang or lang_to_asr

    outputs = [{} for _ in input_file]
    if sli_model:
        sli_outputs = infer(input_file, sli_model, 'sli', sli_out_format, do_vad)
        for out, sli_out in zip(outputs, sli_outputs):
            out.update(**sli_out)
    
    if asr_model and lang_specific_asr:
        for lang, model in lang_to_asr.items():
            if do_vad:
                for file_out in outputs:
                    seg_df = pd.DataFrame(data=file_out['segments'])
                    file = file_out['filename']
                    asr_out = _infer_on_segs(file, seg_df, )
                [out['file'] for out in outputs if out['lang'] == lang]
    elif asr_model:
        ...

    return 

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Annotate an audio file using an ASR and, optionally, AC model.')
    init_annotate_parser(parser)
    args = vars(parser.parse_args(argv))

    handle_annotate(args)
    return 0

if __name__ == '__main__':
    main()