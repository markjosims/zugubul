from zugubul.textnorm import (
    unicode_normalize,
    make_replacements,
    remove_punct,
    strip_diacs,
    get_tone_as_letters,
    split_segs_and_tone,
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

def test_strip_tone():
    text = 'tóňàl ẁōr̂d'
    gold = 'tonal word'
    pred = strip_diacs(text)
    assert pred == gold

def test_get_tone_as_letters():
    text = 'tóňàl ẁōr̂d'
    gold = 'H-LH-L L-M-HL'
    pred = get_tone_as_letters(text)
    assert pred == gold

def test_split_segs_and_tone():
    text = 'tóňàl ẁōr̂d'
    gold = ('tonal word', 'H-LH-L L-M-HL')
    pred = split_segs_and_tone(text)
    assert pred == gold