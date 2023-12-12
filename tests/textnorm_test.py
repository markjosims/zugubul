from zugubul.textnorm import normalize_diac, make_replacements

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