# -*- coding: utf-8 -*-
from torchtext.data import Field
import jieba
import nltk
import os
import torch


class Tokenizer(Field):

    def __init__(self,
                 lang='en',
                 sequential=True,
                 init_token='<sos>',
                 eos_token='<eos>',
                 lower=True,
                 fix_length=64,
                 tokenize=str.split,
                 batch_first=False,
                 **kwargs):

        self.lang = lang
        self.batch_first = batch_first
        tokenize = self.get_tokenizer(lang)
        super(Tokenizer, self).__init__(sequential=sequential, init_token=init_token, eos_token=eos_token, lower=lower,
                                        fix_length=fix_length, tokenize=tokenize, batch_first=batch_first, **kwargs)

    def get_tokenizer(self, lang):
        if lang == 'en':
            tokenize = nltk.word_tokenize
        elif lang == 'zh':
            tokenize = lambda x: jieba.lcut(x, HMM=False)
        else:
            raise ValueError('language not supported')
        return tokenize

    def stoi(self, token):
        if token not in self.vocab.stoi:
            return
        return self.vocab.stoi[token]

    def itos(self, token_id):
        if token_id not in self.vocab.itos:
            return
        return self.vocab.itos[token_id]

    def encode(self, text):
        processed = self.process([text])
        if not self.batch_first:
            token_ids = processed[:, 0].unsqueeze(1)
        else:
            token_ids = processed[0, :].unsqueeze(0)
        return token_ids

    def decode(self, token_ids, sep=None):
        tokens = []
        for token_id in token_ids:
            token = self.vocab.itos[token_id]
            tokens.append(token)
        if isinstance(sep, str):
            return sep.join(tokens)
        return tokens

    def update_props(self):
        self.sos_id = self.stoi(self.init_token)
        self.eos_id = self.stoi(self.eos_token)
        self.pad_id = self.stoi(self.pad_token)

    def build_vocab(self, *args, **kwargs):
        super(Tokenizer, self).build_vocab(*args, **kwargs)
        self.update_props()

    def save(self, path):
        torch.save(self.vocab, path)
        self.update_props()

    def load(self, path):
        self.vocab = torch.load(path)
        self.update_props()
