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
    sli_out_format: Literal['labels', 'probs'] = 'labels',
    task: Literal['asr', 'sli'] = 'asr',
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
        sli_segs = _do_vad_and_infer(input_file, sli_out_format, pipe)
        return sli_segs
    # multiple files w/ VAD
    elif do_vad:
        print('Performing VAD on files...')
        sli_outs = []
        for file in tqdm(input_file):
            sli_outs.extend(_do_vad_and_infer(file, sli_out_format, pipe))
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

def _do_vad_and_infer(
    input_file: Union[str, os.PathLike],
    pipe: Pipeline,
    out_format_funct: Callable,
) -> List[Dict[str, Any]]:
    vad_eaf = label_speech_segments(input_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        vad_df = snip_audio(annotations=vad_eaf, out_dir=tmpdir, audio=input_file)
        vad_segs = vad_df['wav_clip']

        tqdm.write('Performing SLI on VAD segments...')
        sli_out = _do_infer_list(vad_segs, pipe, out_format_funct)

    sli_segs = [{
            'file': input_file,
            'segments': [
                {'start': start, 'end': end, **seg_sli}
                for start, end, seg_sli in zip(vad_df['start'], vad_df['end'], sli_out)
            ]
        }]
    
    return sli_segs

def _do_infer_list(
    input_list: pd.Series,
    pipe: Pipeline,
    out_format_funct: Callable,
):
    out_list = input_list.progress_apply(pipe)
    return out_list.apply(out_format_funct)

def annotate(
        source: Union[str, os.PathLike],
        asr_model: Union[str, os.PathLike, None] = None,
        tgt_lang: Optional[str] = None,
        lid_model: Union[str, os.PathLike, None] = None,
        tier: str = 'default-lt',
        etf: Optional[Union[str, Elan.Eaf]] = None,
        inference_method: Literal['api', 'local'] = 'api'
    ) -> Elan.Eaf:
    """
    Perform LID on source file using model at lid_model (a filepath or URL).
    Remove all annotations not belonging to the target language, then run ASR
    using model at asr_model.
    If tier is provided, add annotations to tier of that name.
    If etf is provided, use as template for output .eaf file.
    """
    if not asr_model and not lid_model:
        raise ValueError("Either ASR or AC model or both must be passed.")

    if lid_model:
        lid_eaf = infer(
            source=source,
            model=lid_model,
            tier=tier,
            etf=etf,
            task='LID',
            inference_method=inference_method,
            return_ac_probs=not asr_model,
        )
        print(len(lid_eaf.get_annotation_data_for_tier(tier)), "speech segments detected from VAD.")
        tgt_eaf = trim(lid_eaf, tier, keepword=tgt_lang)
        print(len(tgt_eaf.get_annotation_data_for_tier(tier)), f"speech segments detected belonging to language {tgt_lang}.")
    else:
        tgt_eaf = label_speech_segments(source)
    if asr_model:
        annotated_eaf = infer(
            source=source,
            model=asr_model,
            eaf=tgt_eaf,
            tier=tier,
            etf=etf,
            task='ASR',
            inference_method=inference_method,
        )
    return annotated_eaf

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Annotate an audio file using an ASR and, optionally, AC model.')
    init_annotate_parser(parser)
    args = vars(parser.parse_args(argv))

    handle_annotate(args)
    return 0

if __name__ == '__main__':
    main()