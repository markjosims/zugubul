from zugubul.textnorm import (
    unicode_normalize,
    make_replacements,
    remove_punct,
    COMBINING
)

ACUTE = COMBINING['acute']
GRAVE = COMBINING['grave']
CARON = COMBINING['caron']
CIRCM = COMBINING['circm']
MACRN = COMBINING['macrn']
TILDE = COMBINING['tilde']

def test_make_replacements():
    # should avoid transitive replacements
    # should replace longer sequences first
    reps = {
        'c': 'k',
        'ts': 'c',
        'a': 'æ',
        'æ': 'aj',
        't': 't̪',
        's': 'ʃʰ',
    }
    text = 'catsæst'
    gold = 'kæcajʃʰt̪'
    pred = make_replacements(text, reps)

    assert pred == gold

def test_unicode_normalize():
    text = "áîèõǔō"
    gold = f"a{ACUTE}i{CIRCM}e{GRAVE}o{TILDE}u{CARON}o{MACRN}"
    pred = unicode_normalize(text)
    assert pred == gold

def test_remove_punct():
    text = 'hi?!?@#$^? ho;<>?[\{\}]w a%*()re you?\'"'
    gold = 'hi how are you'
    pred = remove_punct(text)
    assert pred == gold