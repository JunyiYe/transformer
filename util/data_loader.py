# from torchtext.legacy.data import Field, BucketIterator
# from torchtext.legacy.datasets.translation import Multi30k


# class DataLoader:
#     source: Field = None
#     target: Field = None

#     def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
#         self.ext = ext
#         self.tokenize_en = tokenize_en
#         self.tokenize_de = tokenize_de
#         self.init_token = init_token
#         self.eos_token = eos_token
#         print('dataset initializing start')

#     def make_dataset(self):
#         if self.ext == ('.de', '.en'):
#             self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)
#             self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)

#         elif self.ext == ('.en', '.de'):
#             self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)
#             self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
#                                 lower=True, batch_first=True)

#         train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
#         return train_data, valid_data, test_data

#     def build_vocab(self, train_data, min_freq):
#         self.source.build_vocab(train_data, min_freq=min_freq)
#         self.target.build_vocab(train_data, min_freq=min_freq)

#     def make_iter(self, train, validate, test, batch_size, device):
#         train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
#                                                                               batch_size=batch_size,
#                                                                               device=device)
#         print('dataset initializing done')
#         return train_iterator, valid_iterator, test_iterator


from datasets import load_dataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


class HuggingFaceDataLoader:
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        self.source_vocab = {}
        self.target_vocab = {}
        self.source = None
        self.target = None
        print('dataset initializing start')

    def _tokenize(self, example, source_lang, target_lang):
        source_text = example[source_lang]
        target_text = example[target_lang]

        source_tokens = [self.init_token] + self.tokenize_en(source_text) + [self.eos_token]
        target_tokens = [self.init_token] + self.tokenize_de(target_text) + [self.eos_token]

        return {
            "source_tokens": source_tokens,
            "target_tokens": target_tokens
        }

    def make_dataset(self):
        raw_data = load_dataset("romrawinjp/multi30k")

        source_lang, target_lang = self.ext

        tokenize_fn = lambda x: self._tokenize(x, source_lang[1:], target_lang[1:])  # remove leading dot

        tokenized = raw_data.map(tokenize_fn)

        return tokenized['train'], tokenized['validation'], tokenized['test']

    def build_vocab(self, train_data, min_freq):
        from collections import Counter
        source_counter = Counter()
        target_counter = Counter()

        for example in train_data:
            source_counter.update(example["source_tokens"])
            target_counter.update(example["target_tokens"])

        def build(counter):
            vocab = {"<pad>": 0}
            idx = 1
            for token, freq in counter.items():
                if freq >= min_freq:
                    vocab[token] = idx
                    idx += 1
            return vocab

        self.source_vocab = build(source_counter)
        self.target_vocab = build(target_counter)

    def _numericalize(self, tokens, vocab):
        return [vocab.get(token, vocab.get('<pad>', 0)) for token in tokens]

    def _collate_fn(self, batch):
        source_batch = [torch.tensor(self._numericalize(x["source_tokens"], self.source_vocab)) for x in batch]
        target_batch = [torch.tensor(self._numericalize(x["target_tokens"], self.target_vocab)) for x in batch]

        source_batch = pad_sequence(source_batch, batch_first=True, padding_value=self.source_vocab['<pad>'])
        target_batch = pad_sequence(target_batch, batch_first=True, padding_value=self.target_vocab['<pad>'])

        return source_batch.to(self.device), target_batch.to(self.device)

    def make_iter(self, train, validate, test, batch_size, device):
        self.device = device
        train_iter = TorchDataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        valid_iter = TorchDataLoader(validate, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)
        test_iter = TorchDataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)

        print('dataset initializing done')
        return train_iter, valid_iter, test_iter



