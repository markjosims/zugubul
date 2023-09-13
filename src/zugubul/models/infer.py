from typing import Union, Optional, Literal

import os
import tempfile
import requests
from pympi import Elan
from transformers import pipeline
from huggingface_hub import HfFolder, login

from zugubul.rvad_to_elan import label_speech_segments
from zugubul.models.dataset import process_annotation_length
from zugubul.elan_tools import snip_audio



def query(filename: str, model: str, label_only: bool = False) -> dict:
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
        print("Model still loading, stand by...")
        response = requests.post(api_url, headers=headers, data=data)
    if label_only:
        return get_label_from_query(response.json())
    return response.json()

def get_label_from_query(response_obj: list) -> str:
    max_score = 0
    pred = ''
    for score_dict in response_obj:
        score, label = score_dict['score'], score_dict['label']
        if score > max_score:
            pred = label
    # need to get model to replace "LABEL_n" with ISO code
    # for now replace manually
    id2label = {
        'LABEL_0': 'TIC',
        'LABEL_1': 'ENG',
    }
    return id2label[pred]

def infer(
        source: Union[str, os.PathLike],
        model: Union[str, os.PathLike],
        tier: str = 'default-lt',
        etf: Optional[Union[str, Elan.Eaf]] = None,
        task: Literal['LID', 'ASR'] = 'ASR',
        inference_method: Literal['api', 'local'] = 'api'
    ) -> Elan.Eaf:
    """
    source is a path to a .wav file,
    model is a path (local or url) to a HF LID model, or the model object itself.
    Performs VAD on source, preprocesses 
    """
    eaf = label_speech_segments(
        wav_fp=source,
        tier=tier,
        etf=etf
    )
    trimmed_data = process_annotation_length(eaf)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_data = snip_audio(
            annotations=trimmed_data,
            out_dir=tmpdir,
            audio=source
        )
    
        if inference_method == 'local':
            pipeline_class = 'automatic-speech-recognition' if task=='ASR'\
                else 'audio-classification'
            pipe = pipeline(pipeline_class, model)
            pipe_out = pipe(clip_data['wav_clip'])
            labels = [get_label_from_query(x) for x in pipe_out]
        else:
            query_rows = lambda clip_f: query(clip_f, model, label_only=True)
            labels = clip_data['wav_clip'].apply(query_rows)
        clip_data['text'] = labels


    eaf.remove_all_annotations_from_tier(tier)
    eaf.add_tier(tier)
    add_rows_to_eaf = lambda r: eaf.add_annotation(tier, r['start'], r['end'], r['text'])
    clip_data.apply(add_rows_to_eaf, axis=1)

    return eaf

if __name__ == '__main__':
    f=r'C:\projects\zugubul\tests\wavs\test_tira1.wav'
    model = 'markjosims/wav2vec2-large-mms-1b-tira-lid'
    query(f, model)