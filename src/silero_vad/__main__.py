import torch
from typing import List, Dict

silero_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def run_vad(wav: str) -> List[Dict[str, int]]:
    audio = read_audio(wav)
    return get_speech_timestamps(audio, silero_vad)