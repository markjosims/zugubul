from zugubul.error_analysis import get_edit_dict

def test_edit_dict():
    label = 'apri jicilo'
    pred = 'apr jicilo'
    d = get_edit_dict(label, pred)
    assert d == {'i': {'del': 1}}

def test_edit_dict1():
    label = 'apri jicilo'
    pred = 'aprii jicilo'
    d = get_edit_dict(label, pred)
    assert d == {'i': {'ins': 1}}

def test_edit_dict2():
    label = 'apri jicilo'
    pred = 'apro jicilo'
    d = get_edit_dict(label, pred)
    assert d == {'i': {'rep': {'o':1} } }

def test_edit_dict3():
    label = 'apri jicilo'
    pred = 'apro jciloi'
    d = get_edit_dict(label, pred)
    assert d == {
        'i': {'del': 1, 'ins': 1, 'rep': {'o':1} }
    }
