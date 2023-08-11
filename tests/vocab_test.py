import csv
import json
import os

from zugubul.models.vocab import tokenizer_from_csv, tokenizer_from_list, vocab_from_list

def test_vocab_from_list(tmp_path):
    vocab_dir = tmp_path / 'vocab1'
    vocab_dir.mkdir()
    vocab = [
        '[ENG]',
        '[TIC]',
        '[DDN]',
    ]
    json_path = vocab_from_list(vocab, vocab_dir)
    with open(json_path) as f:
        json_vocab = json.load(f)
    assert json_vocab == {
        '[ENG]': 0,
        '[TIC]': 1,
        '[DDN]': 2,
    }


def test_tokenizer_from_list(tmp_path):
    vocab = [
        '[ENG]',
        '[TIC]',
        '[DDN]',
    ]
    vocab_dir = tmp_path / 'vocab2'
    vocab_dir.mkdir()
    tokenizer = tokenizer_from_list(vocab=vocab, vocab_dir=vocab_dir)
    assert tokenizer.encode('[TIC][ENG][ENG][DDN][TIC]') == [1, 0, 0, 2, 1]


def test_tokenizer_from_csv(tmp_path):
    vocab_dir = tmp_path / 'vocab3'
    vocab_dir.mkdir()
    csv_path = os.path.join(vocab_dir, 'vocab.csv')
    csv_vocab = [
        ['[ENG]', 'foo.wav', 0, 1],
        ['[TIC]', 'foo.wav', 2, 3],
        ['[DDN]', 'bar.wav', 0, 1],
        ['[ENG]', 'bar.wav', 4, 5],
    ]
    with open(csv_path, 'w') as f:
        vocab_writer = csv.writer(f, delimiter='\t', dialect='unix')
        vocab_writer.writerows(csv_vocab)
    tokenizer = tokenizer_from_csv(csv_path=csv_path, vocab_dir=vocab_dir)
    # because tokenizer_from_csv puts the token in a set, we can't know for certain what the indices will be
    # for that reason we're not checking the encoded values here
    assert len(tokenizer.encode('[TIC][ENG][ENG][DDN][TIC]')) == 5

def test_special_tokens(tmp_path):
    vocab = [
        '[ENG]',
        '[TIC]',
        '[DDN]',
    ]
    vocab_dir = tmp_path / 'vocab2'
    vocab_dir.mkdir()
    tokenizer = tokenizer_from_list(vocab=vocab, vocab_dir=vocab_dir)
    assert tokenizer.pad_token == '<pad>'
    assert tokenizer.unk_token == '<unk>'
    assert tokenizer.word_delimiter_token == '|'
