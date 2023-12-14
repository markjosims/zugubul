import PySimpleGUIWx as sg
from typing import Dict
from zugubul.textnorm import get_char_metadata

def char_metadata_row(row_num: int, char_obj: Dict[str, Dict[str, str]]):
    return [
        [sg.Text(f"{row_num}: Character `{char_obj['character']}`"), sg.Text(f"{char_obj['unicode_name']}"), sg.Text(f"Unicode point: {char_obj['unicode_point']}")],
        [sg.Text("Replace with:"), sg.Input(key=row_num, size=(5,1))]
    ]

def multichar_row(row_num: int):
    return [
        [sg.Text("Replace sequence:"), sg.Input(key=row_num, size=(5,1)), sg.Text("Replace with:"), sg.Input(key=row_num, size=(5,1))]
    ]

def char_metadata_window(char_metadata: Dict[str, Dict[str, Dict[str, str]]]):
    rows = [
        [sg.Text(
            "For each character, leave the text field blank to leave the character as is, "+\
            "or enter a character to replace it with."
        )],
        [sg.Text(
            "Type 'None' to delete the character from the dataset."
        )],
    ]
    for i, char_obj in enumerate(char_metadata):
        rows.extend(char_metadata_row(i+1, char_obj))
    rows.extend([
        [sg.Text('Define rules replacing multiple characters.'),],
        [sg.Text('Write rules as in:out,in:out,in:out...'),],
        [sg.Text('For example tʃ:tʂ,kj:c will change all instances of `tʃ` to `tʂ` and `kj` to `c`.'),],
        [sg.Text('Use `None` to delete a sequence, e.g. əʔ:None will delete `əʔ`.'),],
        [sg.Input(key='multichar')],
        [sg.Submit()],
    ])

    # wrap rows in scrollable column
    layout = [
    [
        sg.Column(rows, scrollable=True,  vertical_scroll_only=True),
    ]
]
    window=sg.Window('Character normalization', layout)
    event, values = window.Read()

    if event == sg.WINDOW_CLOSED or event == 'Quit':
        window.close()
        return

    reps = {}
    for i, char_obj in enumerate(char_metadata):
        char = char_obj['character']
        rep = values[i+1]
        if not rep:
            continue
        if rep.lower() == 'none':
            rep = ''
            print(f"Removing all instances of char `{char}`")
        else:
            print(f"Replacing char `{char}` with `{rep}`")
        reps[char] = rep
    multichar_rules = values['multichar']
    for rule in multichar_rules.split(','):
        rule = rule.strip()
        intab, outtab = rule.split(':')
        if outtab.lower() == 'none':
            print(f"Removing all instances of sequence `{intab}`")
            reps[intab] = ''
        else:
            reps[intab] = outtab
            print(f"Replacing sequence `{intab}` with `{outtab}`")


    return reps

if __name__ == '__main__':
    print(char_metadata_window(get_char_metadata(['hellɔo', 'goodbye'])))