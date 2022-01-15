# -*- coding: utf-8 -*-
from torchtext.data import Dataset as TorchDataset
from torchtext.data import Example


class Dataset(TorchDataset):

    def __init__(self, sources, targets, tokenizer4src, tokenizer4tgt):
        fields = [('source', tokenizer4src), ('target', tokenizer4tgt)]
        examples = []
        for source, target in zip(sources, targets):
            example = Example.fromlist([source, target], fields)
            examples.append(example)
        super(Dataset, self).__init__(examples, fields)

