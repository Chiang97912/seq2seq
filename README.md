# seq2seq
Industrial-grade implementation of seq2seq algorithm based on Pytorch, integrated beam search algorithm.

Highlights:
1. easy to train, predict and deploy;
2. lightweight implementation;
3. multitasking support (including dialogue generation and machine translation).

## Model description

* Encoder: Bidirectional GRU
* Decoder: GRU with Attention Mechanism
* Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

## Dependencies

* `python`  version 3.6
* `pyTorch`  version 1.9.0
* `torchtext`  version 0.3.1
* `numpy`  version 1.19.5
* `nltk`  version 3.5
* `jieba`  version 0.42.1


## References

* [PyTorch Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [@spro/practical-pytorch](https://github.com/spro/practical-pytorch)
* [@AuCson/PyTorch-Batch-Attention-Seq2seq](https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq)
* [@keon/seq2seq](https://github.com/keon/seq2seq)

