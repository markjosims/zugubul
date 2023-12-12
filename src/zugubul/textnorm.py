from typing import Dict

combining = {
    'grave': "\u0300",
    'macrn': "\u0304",
    'acute': "\u0301",
    'circm': "\u0302",
    'caron': "\u030C"
}

composite = {
    "a": {"acute": "á", "macrn": "ā", "grave": "à", "caron": "ǎ", "circm": "â"},
    "e": {"acute": "é", "macrn": "ē", "grave": "è", "caron": "ě", "circm": "ê"},
    "i": {"acute": "í", "macrn": "ī", "grave": "ì", "caron": "ǐ", "circm": "î"},
    "o": {"acute": "ó", "macrn": "ō", "grave": "ò", "caron": "ǒ", "circm": "ô"},
    "u": {"acute": "ú", "macrn": "ū", "grave": "ù", "caron": "ǔ", "circm": "û"},
}

def normalize_diac(text: str) -> str:
    ...

def make_replacements(text: str, reps: Dict[str, str]) -> str:
    ...