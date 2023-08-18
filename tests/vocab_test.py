import tempfile
import pytest
import csv
import json
import os

from zugubul.models.vocab import vocab_from_csv, vocab_from_list, init_processor
from transformers import Wav2Vec2CTCTokenizer


@pytest.fixture
def tmp_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_dir = os.path.join(tmpdir, 'tmp_vocab')
        os.mkdir(vocab_dir)
        csv_path = os.path.join(vocab_dir, 'vocab.csv')
        csv_vocab = [
            ['lang', 'file_name', 'start', 'end'],
            ['[ENG]', 'foo.wav', 0, 1],
            ['[TIC]', 'foo.wav', 2, 3],
            ['[DDN]', 'bar.wav', 0, 1],
            ['[ENG]', 'bar.wav', 4, 5],
        ]
        with open(csv_path, 'w') as f:
            vocab_writer = csv.writer(f, delimiter='\t', dialect='unix')
            vocab_writer.writerows(csv_vocab)
        yield csv_path, vocab_dir

def test_vocab_from_list(tmp_path):
    vocab_dir = tmp_path / 'vocab1'
    vocab_dir.mkdir()
    vocab = [
        '[ENG]',
        '[TIC]',
        '[DDN]',
    ]
    json_path = vocab_from_list(vocab, vocab_dir, lid=True)
    with open(json_path) as f:
        json_vocab = json.load(f)
    assert json_vocab['[ENG]'] != json_vocab['[TIC]']
    assert json_vocab['[DDN]'] != json_vocab['[TIC]']
    assert json_vocab['[ENG]'] != json_vocab['[DDN]']


def test_tokenizer_from_list(tmp_path):
    vocab = [
        '[ENG]',
        '[TIC]',
        '[DDN]',
    ]
    vocab_dir = tmp_path / 'vocab2'
    vocab_dir.mkdir()
    vocab = vocab_from_list(vocab=vocab, vocab_dir=vocab_dir, lid=True)
    tokenizer = Wav2Vec2CTCTokenizer(vocab)
    assert len(tokenizer.encode('[TIC][ENG][ENG][DDN][TIC]')) == 5
    assert tokenizer.encode('[TIC]') != tokenizer.encode('[ENG]')
    assert tokenizer.encode('[DDN]') != tokenizer.encode('[ENG]')
    assert tokenizer.encode('[DDN]') != tokenizer.encode('[TIC]')


def test_tokenizer_from_csv(tmp_csv):
    csv_path, vocab_dir = tmp_csv
    vocab = vocab_from_csv(csv_path=csv_path, vocab_dir=vocab_dir, lid=True)
    tokenizer = Wav2Vec2CTCTokenizer(vocab)
    # because vocab_from_csv puts the tokens in a set, we can't know for certain what the indices will be
    # for that reason we're not checking the encoded values here
    assert len(tokenizer.encode('[TIC][ENG][ENG][DDN][TIC]')) == 5
    assert tokenizer.encode('[TIC]') != tokenizer.encode('[ENG]')
    assert tokenizer.encode('[DDN]') != tokenizer.encode('[ENG]')
    assert tokenizer.encode('[DDN]') != tokenizer.encode('[TIC]')

def test_special_tokens(tmp_path):
    vocab = [
        '[ENG]',
        '[TIC]',
        '[DDN]',
    ]
    vocab_dir = tmp_path / 'vocab4'
    vocab_dir.mkdir()
    vocab = vocab_from_list(vocab=vocab, vocab_dir=vocab_dir, lid=True)
    tokenizer = Wav2Vec2CTCTokenizer(vocab)
    assert tokenizer.pad_token == '<pad>'
    assert tokenizer.unk_token == '<unk>'
    assert tokenizer.word_delimiter_token == '|'

def test_init_processor(tmp_path):
    vocab = [
        '[ENG]',
        '[TIC]',
        '[DDN]',
    ]
    vocab_dir = tmp_path / 'vocab5'
    vocab_dir.mkdir()
    processor = init_processor(vocab=vocab, vocab_dir=vocab_dir, lid=True)
    assert len(processor(text='[ENG][TIC][DDN]').input_ids) == 3
    assert processor(text='[DDN]').input_ids != processor(text='[TIC]').input_ids
    assert processor(text='[DDN]').input_ids != processor(text='[ENG]').input_ids
    assert processor(text='[ENG]').input_ids != processor(text='[TIC]').input_ids
