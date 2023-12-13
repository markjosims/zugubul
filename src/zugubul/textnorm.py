from typing import Dict, Sequence, Any
from string import punctuation

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

def max_ord_in_str(text: str) -> int:
    return max(ord(c) for c in text)

def make_replacements(text: str, reps: Dict[str, str]) -> str:
    """
    Makes all replacements specified by `reps`, a dict whose keys are intabs
    and values are outtabs to replace them.
    Avoids transitivity by first replacing intabs to a unique char not found in the original string.
    """
    max_ord = max_ord_in_str(text)
    intab2unique = {
        k: chr(max_ord+i+1) for i, k in enumerate(reps.keys())
    }
    unique2outtab = {
        intab2unique[k]: v for k, v in reps.items()
    }

    # sort intabs so that longest sequences come first
    intabs = sorted(reps.keys(), key=len, reverse=True)

    for intab in intabs:
        sentinel = intab2unique[intab]
        text = text.replace(intab, sentinel)
    for sentinel, outtab in unique2outtab.items():
        text = text.replace(sentinel, outtab)

    return text

def remove_punct(text: str) -> str:
    for p in punctuation:
        text = text.replace(p, '')
    return text

def report_unique_chars(texts: Sequence[str]) -> Dict[str, Any]:
    unique = set()
    (unique.update(text) for text in texts)
    # find some way to get Unicode metadata for each character