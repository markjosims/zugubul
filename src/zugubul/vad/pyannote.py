from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from typing import Optional
from torchaudio import load


def run_pyannote_vad(
        wav_fp: str,
        pipe: Optional[PyannotePipeline] = None,
        sample_rate: int = 16000,
    ):
    wav = AudioSegment.from_file(wav_fp)
    wav = wav.set_frame_rate(sample_rate)

    if not pipe:
        pipe = PyannotePipeline.from_pretrained("pyannote/voice-activity-detection")

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": wav, "sample_rate": sample_rate},
            hook=hook,
        )
    return result