from pympi import Elan
from typing import Literal
import sys

def read_rvad_segs(fp: str, dialect: Literal['seg', 'frame']='seg') -> list:
    if dialect == 'seg':
        with open(fp, 'r') as f:
            segs = f.readlines()
        segs = [int(s) for s in segs]
        startpoints = segs[:len(segs)//2]
        endpoints = segs[len(segs)//2:]
    else:
        # dialect == 'frame'
        with open(fp, 'r') as f:
            frames = f.readlines()
        frames = [int(f) for f in frames]
        in_seg = False
        startpoints = []
        endpoints = []
        for i, f in enumerate(frames):
            if in_seg:
                if f == 0:
                    # end of segment
                    in_seg = False
                    endpoints.append(i)
            elif f == 1:
                # beginning of segment
                in_seg = True
                startpoints.append(i+1)
    return [(start, end) for start, end in zip(startpoints, endpoints)]


def rvad_segs_to_ms(segs: list) -> list:
    frame_width = 10
    return [(start*frame_width, end*frame_width) for start, end in segs]

def make_annotation(eaf: Elan.Eaf, start: int, end: int):
    eaf.add_annotation('default-lt', start, end)

def main():
    rvad_fp = sys.argv[1]
    eaf_fp = sys.argv[2]
    segs = read_rvad_segs(rvad_fp)
    times = rvad_segs_to_ms(segs)
    eaf = Elan.Eaf()
    eaf.add_tier('default-lt')
    for start, end in times:
        make_annotation(eaf, start, end)
    Elan.to_eaf(eaf_fp, eaf)
    

if __name__ == '__main__':
    main()