from typing import Union, Optional, Literal, Sequence, List, Dict

import os
import tempfile
from pympi import Elan
from tqdm import tqdm
from transformers import pipeline

from zugubul.vad_to_elan import label_speech_segments
from zugubul.models.dataset import process_annotation_length
from zugubul.elan_tools import snip_audio, trim
from zugubul.main import init_annotate_parser, handle_annotate
import argparse

# enable pandas progress bars
tqdm.pandas()

def pipe_out_to_label(response_obj: List[Dict[str, str]]) -> str:
    max_score = 0
    pred = ''
    for score_dict in response_obj:
        score, label = score_dict['score'], score_dict['label']
        if score > max_score:
            pred = label
            max_score = score
    return pred

def pipe_out_to_probs(response_obj: List[Dict[str, str]]) -> Dict[str, str]:
    scores = {}
    for score_dict in response_obj:
        score, label = score_dict['score'], score_dict['label']
        scores[label] = score
    return scores

def infer(
        source: Union[str, os.PathLike],
        model: Union[str, os.PathLike],
        tier: Optional[str] = None,
        eaf: Union[str, os.PathLike, Elan.Eaf, None] = None,
        etf: Union[str, os.PathLike, Elan.Eaf, None] = None,
        task: Literal['LID', 'ASR'] = 'ASR',
        tgt_lang: Optional[str] = None,
        return_ac_probs: bool = False,
        max_len: int = 5,
    ) -> Elan.Eaf:
    """
    source is a path to a .wav file,
    model is a path (local or url) to a HF LID model, or the model object itself.
    tier is a str indicating which tier to add annotations to in .eaf file.
    eaf is an Eaf obj or a path to a .eaf file.
    If eaf is not passed, VAD is performed on source and the output is used for annotation.
    etf is an Eaf object of a .etf template file or a path to a .etf file.
    task is a str indicating whether language identification or automatic speech recognition is being performed.
    """
    if not eaf:
        print("Performing VAD...")
        eaf = label_speech_segments(
            wav_fp=source,
            tier=tier,
            etf=etf
        )
        #data = process_annotation_length(eaf)
    else:
        if type(eaf) is not Elan.Eaf:
            eaf = Elan.Eaf(eaf)
    
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_data = snip_audio(
            annotations=eaf,
            out_dir=tmpdir,
            audio=source,
            tier=tier,
        )
        print(f"VAD detected {len(clip_data)} speech segments in source.")
        above_max_len = clip_data.apply(lambda r: r['end']-r['start'] > max_len*1000, axis=1)
        clip_data = clip_data[~above_max_len]
        print(f"After removing segments longer than {max_len} seconds {len(clip_data)} segments remain.")

        print(f"Running inference for {task} using {model}...")

        pipeline_class = 'automatic-speech-recognition' if task=='ASR'\
            else 'audio-classification'
        pipe = pipeline(pipeline_class, model)
        pipe_out = clip_data['wav_clip'].progress_apply(pipe)
        if task == 'LID' and return_ac_probs:
            labels = [pipe_out_to_probs(x) for x in pipe_out]
        elif task == 'LID':
            labels = [pipe_out_to_label(x) for x in pipe_out]
        else:
            labels = [x['text'] for x in pipe_out]

        clip_data['text'] = labels

    if return_ac_probs:
            tiers = labels[0].keys()
            if tgt_lang:
                tiers = [tgt_lang,]
            for t in tiers:
                eaf.add_tier(t)
            def add_probs_to_eaf(row):
                probs = row['text']
                start = row['start']
                end = row['end']
                for t in tiers:
                    eaf.add_annotation(t, start, end, str(round(probs[t], 5)))
            clip_data.apply(add_probs_to_eaf, axis=1)
            return eaf
        

    tier = 'default-lt' if not tier else tier
    eaf.remove_all_annotations_from_tier(tier)
    eaf.add_tier(tier)
    add_rows_to_eaf = lambda r: eaf.add_annotation(tier, r['start'], r['end'], r['text'])
    clip_data.apply(add_rows_to_eaf, axis=1)

    return eaf

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