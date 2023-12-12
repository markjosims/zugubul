from typing import Dict

DIACS = ['grave', 'macrn', 'acute', 'circm', 'caron', 'tilde',]

COMBINING = {
    'grave': "\u0300",
    'macrn': "\u0304",
    'acute': "\u0301",
    'circm': "\u0302",
    'caron': "\u030C",
    'tilde': "\u0303",
}
COMPOSITE = {
    "a": {"acute": "á", "macrn": "ā", "grave": "à", "caron": "ǎ", "circm": "â", "tilde": "ã",},
    "e": {"acute": "é", "macrn": "ē", "grave": "è", "caron": "ě", "circm": "ê", "tilde": "ẽ",},
    "i": {"acute": "í", "macrn": "ī", "grave": "ì", "caron": "ǐ", "circm": "î", "tilde": "ĩ",},
    "o": {"acute": "ó", "macrn": "ō", "grave": "ò", "caron": "ǒ", "circm": "ô", "tilde": "õ",},
    "u": {"acute": "ú", "macrn": "ū", "grave": "ù", "caron": "ǔ", "circm": "û", "tilde": "ũ",},
}

def diac_to_combining(text: str) -> str:
    for basechar, composite_dict in COMPOSITE.items():
        for diac, composite_char in composite_dict.items():
            text = text.replace(composite_char, basechar+COMBINING[diac])
    
    return text

def make_replacements(text: str, reps: Dict[str, str]) -> str:
    intab2sentinel = {
        k: object() for k in reps.keys()
    }
    sentinel2outtab = {
        intab2sentinel[k]: v for k, v in reps.items()
    }

    # sort intabs so that longest sequences come first
    intabs = sorted(reps.keys(), key=len, reverse=True)

    for intab in intabs:
        sentinel = intab2sentinel[intab]
        text = text.replace(intab, sentinel)
    for sentinel, outtab in sentinel2outtab.items():
        text = text.replace(sentinel, outtab)

    return text