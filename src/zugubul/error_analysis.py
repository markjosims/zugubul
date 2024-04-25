import visedit
from visedit.utils import Levenshtein as Lev
import argparse
from typing import Optional, Sequence, Dict, Union, Any
import json
from collections import defaultdict
from zugubul.textnorm import split_segs_and_tone

def edit_dict_factory():
    return defaultdict(lambda: {
        'ins': 0,
        'del': 0,
        'rep': defaultdict(lambda: 0)
    })

# make dictionary counting edits for each possible char>char edit
def get_edit_dict(label: str, pred: str) -> Dict[str, Dict[str, Union[int, Dict[str, int]]]]:
    m = Lev.leven(label, pred)
    path = Lev.find_path(m)
    edits = edit_dict_factory()

    label_i = 0
    pred_i = 0
    for edit in path:
        pred_char = pred[pred_i] if pred_i < len(pred) else None
        label_char = label[label_i] if label_i < len(label) else None

        if edit == 'ins':
            edits[pred_char]['ins']+=1
            pred_i+=1
        elif edit == 'del':
            edits[label_char]['del']+=1
            label_i+=1
        elif edit == 'rep':
            edits[label_char]['rep'][pred_char]+=1
            label_i+=1
            pred_i+=1
        else:
            label_i+=1
            pred_i+=1
        
    return clean_edit_dict(edits)

def merge_edit_dicts(
        main_dict: Dict[str, Dict[str, Any]],
        incoming: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
    for char, edits in incoming.items():
        for edit, val in edits.items():
            if type(val) is int:
                main_dict[char][edit]+=val
            else:
                for tgt_char, ct in val.items():
                    main_dict[char][edit][tgt_char]+=ct
    
    return main_dict

def clean_edit_dict(d: Dict[str, Dict[str, Any]]):
    for char, edits in d.items():
        for edit, val in list(edits.items()):
            if (val==0) or (edit=='rep' and len(val)==0):
                d[char].pop(edit)
                
    
    return d

def add_prefixes_to_html_chunk(html_chunk: str) -> str:
    html_chunk = html_chunk.replace(
        '<tr>', '<tr><td>Label:</td>', 1
    )
    html_chunk = html_chunk.replace(
        '</tr><tr>',
        '</tr><tr><td>Predicted:</td>'
    ) 
    return html_chunk

def init_error_parser(error_parser: argparse.ArgumentParser) -> None:
    error_parser.add_argument('IN')
    error_parser.add_argument('--html', '-H')
    error_parser.add_argument('--json', '-j')

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = argparse.ArgumentParser('Generate visualization of error from preds.json file.')
    init_error_parser(parser)

    args = parser.parse_args(argv)

    with open(args.IN) as f:
        data = json.load(f)
    
    html_chunks = []
    seg_edits = edit_dict_factory()
    tone_edits = edit_dict_factory()
    for record in data:
        pred = record['pred']
        label = record['label']
        audio_path = record['audio_path']

        pred_segs, pred_tone = split_segs_and_tone(pred)
        label_segs, label_tone = split_segs_and_tone(label)

        # update html doc
        html_chunks.append(f"Record {record['i']}:")
        get_edit_html(html_chunks, pred, label)
        html_chunks.append('Segments')
        get_edit_html(html_chunks, pred_segs, label_segs)
        html_chunks.append('Tones')
        get_edit_html(html_chunks, pred_tone, label_tone)

        add_html_audio(html_chunks, audio_path)

        # update edit object
        incoming_seg_edits = get_edit_dict(pred=pred_segs, label=label_segs)
        merge_edit_dicts(seg_edits, incoming_seg_edits)

        incoming_tone_edits = get_edit_dict(pred=pred_tone, label=label_tone)
        merge_edit_dicts(tone_edits, incoming_tone_edits)

    seg_edits = clean_edit_dict(seg_edits)
    tone_edits = clean_edit_dict(tone_edits)
    edits = {'seg_edits': seg_edits, 'tone_edits': tone_edits}

    with open(args.html, 'w') as f:
        f.writelines(html_chunks)
    
    with open(args.json, 'w') as f:
        json.dump(edits, f, indent=2, ensure_ascii=False)


    return 0

def get_edit_html(
        html_chunks: list,
        pred: str,
        label: str,
    ):
    se = visedit.StringEdit(target_str=pred, source_str=label)
    html_chunk = se.generate_html()
    html_chunk = add_prefixes_to_html_chunk(html_chunk)
    html_chunks.append(html_chunk)

def add_html_audio(html_chunks: list, audio_path: str):
    html_chunks.append(
        f'''
        <audio controls>
            <source src="{audio_path}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        '''
    )

if __name__ == '__main__':
    main()