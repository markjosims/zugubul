from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from typing import Optional, Literal
import torchaudio

def run_pyannote_vad(
        wav_fp: str,
        pipe: Optional[PyannotePipeline] = None,
        sample_rate: int = 16000,
        jsonify: bool = True,
        time_format: Literal['ms', 'sec', 'samples'] = 'ms',
    ):
    wav_orig, sr_orig = torchaudio.load(wav_fp)
    wav = torchaudio.functional.resample(
        waveform=wav_orig,
        orig_freq=sr_orig,
        new_freq=sample_rate
    )

    if not pipe:
        pipe = PyannotePipeline.from_pretrained("pyannote/voice-activity-detection")

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": wav, "sample_rate": sample_rate},
            hook=hook,
        )

    if jsonify:
            segments = []
            for track, _ in result.itertracks():
                start, end = track.start, track.end
                if time_format == 'ms':
                    start = int(start*1000)
                    end = int(end*1000)
                elif time_format == 'samples':
                    start = int(start*sample_rate)
                    end = int(end*sample_rate)
                segment = {'start': start, 'end': end}
                segments.append(segment)
            return segments
    
    return result
