from typing import Union, Optional, Literal, Sequence, List

import os
import tempfile
import requests
from pympi import Elan
from time import sleep
from transformers import pipeline
from huggingface_hub import HfFolder, login

from zugubul.rvad_to_elan import label_speech_segments
from zugubul.models.dataset import process_annotation_length
from zugubul.elan_tools import snip_audio, trim



def query(filename: str, model: str, label_only: bool = False, task: Literal['ASR', 'LID'] = 'ASR') -> dict:
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    token = HfFolder.get_token()
    while not token:
        login()
        token = HfFolder.get_token()

    headers = {"Authorization": f"Bearer {token}"}
    with open(filename, "rb") as f:
        data = f.read()

    response = requests.post(api_url, headers=headers, data=data)
    while response.status_code == 503:
        wait_time = response.json()['estimated_time']
        print(f"Model still loading, waiting for {wait_time} seconds...")
        sleep(wait_time)
        response = requests.post(api_url, headers=headers, data=data)
    print("Fetched query.")
    if label_only:
        if task == 'LID':
            return get_label_from_query(response.json())
        return response.json()['text']
    return response.json()

def query_list(
        filename: Sequence[str],
        model: str,
        task: Literal['ASR', 'LID'] = 'ASR',
        label_only: bool = False,
    ) -> List[Union[str, dict[str, str]]]:
    # only build pipeline if API query returns error
    pipe = None
    outputs = []
    for f in filename:
        if pipe:
            response_obj = pipe(f)
        else:
            response_obj = query(f, model)
            if 'error' in response_obj:
                print("API returned error, building local pipeline...")
                pipeline_class = 'automatic-speech-recognition' if task=='ASR'\
                    else 'audio-classification'
                pipe = pipeline(pipeline_class, model)
                response_obj = pipe(f)

        if label_only and (task=='LID'):
            outputs.append(get_label_from_query(response_obj))
        elif label_only and (task=='ASR'):
            outputs.append(response_obj['text'])
        else:
            outputs.append(response_obj)
    
    return outputs

def get_label_from_query(response_obj: list) -> str:
    max_score = 0
    pred = ''
    for score_dict in response_obj:
        score, label = score_dict['score'], score_dict['label']
        if score > max_score:
            pred = label
            max_score = score
    return pred

def infer(
        source: Union[str, os.PathLike],
        model: Union[str, os.PathLike],
        tier: str = 'default-lt',
        eaf: Union[str, os.PathLike, Elan.Eaf, None] = None,
        etf: Union[str, os.PathLike, Elan.Eaf, None] = None,
        task: Literal['LID', 'ASR'] = 'ASR',
        inference_method: Literal['api', 'local', 'try_api'] = 'try_api'
    ) -> Elan.Eaf:
    """
    source is a path to a .wav file,
    model is a path (local or url) to a HF LID model, or the model object itself.
    tier is a str indicating which tier to add annotations to in .eaf file.
    eaf is an Eaf obj or a path to a .eaf file.
    If eaf is not passed, VAD is performed on source and the output is used for annotation.
    etf is an Eaf object of a .etf template file or a path to a .etf file.
    task is a str indicating whether language identification or automatic speech recognition is being performed.
    inference_method is a str indicating whether the HuggingFace api should be used for inference,
    or whether the model should be downloaded for local inference.
    """
    if not eaf:
        print("Performing VAD...")
        eaf = label_speech_segments(
            wav_fp=source,
            tier=tier,
            etf=etf
        )
        data = process_annotation_length(eaf)
    else:
        if type(eaf) is not Elan.Eaf:
            eaf = Elan.Eaf(eaf)
        data = eaf
    
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_data = snip_audio(
            annotations=data,
            out_dir=tmpdir,
            audio=source
        )

        print(f"Running inference for {task}...")
        if inference_method == 'local':
            pipeline_class = 'automatic-speech-recognition' if task=='ASR'\
                else 'audio-classification'
            pipe = pipeline(pipeline_class, model)
            pipe_out = clip_data['wav_clip'].apply(pipe)
            if task == 'LID':
                labels = [get_label_from_query(x) for x in pipe_out]
            else:
                labels = [x['text'] for x in pipe_out]
        elif inference_method == 'api':
            query_rows = lambda clip_f: query(clip_f, model, label_only=True)
            labels = clip_data['wav_clip'].apply(query_rows)
        else:
            # inference_method == 'try_api'
            labels = query_list(
                clip_data['wav_clip'],
                model,
                task,
                label_only=True
            )
        clip_data['text'] = labels


    eaf.remove_all_annotations_from_tier(tier)
    eaf.add_tier(tier)
    add_rows_to_eaf = lambda r: eaf.add_annotation(tier, r['start'], r['end'], r['text'])
    clip_data.apply(add_rows_to_eaf, axis=1)

    return eaf

def annotate(
        source: Union[str, os.PathLike],
        lid_model: Union[str, os.PathLike],
        asr_model: Union[str, os.PathLike],
        tgt_lang: str,
        tier: str = 'default-lt',
        etf: Optional[Union[str, Elan.Eaf]] = None,
        inference_method: Literal['api', 'local'] = 'api'
    ) -> Elan.Eaf:
    lid_eaf = infer(
        source=source,
        model=lid_model,
        tier=tier,
        etf=etf,
        task='LID',
        inference_method=inference_method,
    )
    print(len(lid_eaf.get_annotation_data_for_tier(tier)), "speech segments detected from VAD.")
    tgt_eaf = trim(lid_eaf, tier, keepword=tgt_lang)
    print(len(tgt_eaf.get_annotation_data_for_tier(tier)), f"speech segments detected belonging to language {tgt_lang}.")
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