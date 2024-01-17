import torch
from typing import List, Dict

SAMPLE_RATE = 16000

silero_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def run_silero_vad(wav_fp: str, convert_to_ms: bool = True) -> List[Dict[str, int]]:
    """
    Run silero_vad on the file indicated by `wav`.
    silero_vad returns sample indices by default,
    this method converts samples into ms.
    Pass `convert_to_ms=False` to keep the original behavior and pass samples.
    """
    audio = read_audio(wav_fp, sampling_rate=SAMPLE_RATE)
    timestamps = get_speech_timestamps(audio, silero_vad, sampling_rate=SAMPLE_RATE, return_seconds=True)
    if convert_to_ms:
        for seg in timestamps:
            seg['start']=int(seg['start']*1000)
            seg['end']=int(seg['end']*1000)

    return timestamps