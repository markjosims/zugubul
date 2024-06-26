import torch
from typing import List, Dict, Callable, Literal

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


def run_silero_vad(
        wav_fp: str,
        time_format: Literal['ms', 'sec', 'samples'] = 'ms',
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        visualize_probs: bool = False,
        progress_tracking_callback: Callable[[float], None] = None,
    ) -> List[Dict[str, int]]:
    """
    Run silero_vad on the file indicated by `wav`.
    `time_format' determines the format for representing timestamps,
    `ms` for milliseconds, `sec` for seconds, `samples` for sample
    indices.
    Returns list of dicts:
    [
        {'start': TIME, 'end': TIME},
        {'start': TIME, 'end': TIME},
        ...
    ]
    """
    audio = read_audio(
        wav_fp,
        sampling_rate=sampling_rate or SAMPLE_RATE,
    )
    vad_kwargs = {
        'threshold':                    threshold,
        'min_speech_duration_ms':       min_speech_duration_ms,
        'max_speech_duration_s':        max_speech_duration_s,
        'min_silence_duration_ms':      min_silence_duration_ms,
        'window_size_samples':          window_size_samples,
        'speech_pad_ms':                speech_pad_ms,
        'visualize_probs':              visualize_probs,
        'progress_tracking_callback':   progress_tracking_callback,
        'return_seconds':               time_format != 'samples',
    }

    timestamps = get_speech_timestamps(
        audio,
        silero_vad,
        sampling_rate=sampling_rate or SAMPLE_RATE,
        **vad_kwargs,
    )
    if time_format == 'ms':
        for seg in timestamps:
            seg['start']=int(seg['start']*1000)
            seg['end']=int(seg['end']*1000)

    return timestamps