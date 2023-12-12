from zugubul.textnorm import diac_to_combining, make_replacements, COMBINING

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

def test_diacs_to_combining():
    text = "áîèõǔō"
    gold = f"a{ACUTE}i{CIRCM}e{GRAVE}o{TILDE}u{CARON}o{MACRN}"
    pred = diac_to_combining(text)
    assert gold == pred