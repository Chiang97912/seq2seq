# seq2seq
Industrial-grade implementation of seq2seq algorithm based on Pytorch, integrated beam search algorithm.

seq2seq is based on other excellent open source projects, this project has the following highlights:
1. easy to train, predict and deploy;
2. lightweight implementation;
3. multitasking support (including dialogue generation and machine translation).



## Model description

* Encoder: Bidirectional GRU
* Decoder: GRU with Attention Mechanism
* Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)



## Install

seq2seq is dependent on PyTorch. Two ways to install:

**Install seq2seq from Pypi:**

```
pip install seq2seq-pytorch
```



**Install seq2seq from the Github source:**

```
git clone https://github.com/Chiang97912/seq2seq.git
cd seq2seq
python setup.py install
```



## Usage

### Train

```python
from seq2seq.model import Seq2Seq

sources = ['...']
targets = ['...']
model = Seq2Seq('seq2seq-model', embed_size=256, hidden_size=512, lang4src='en', lang4tgt='en', device='cuda:0')
model.fit(sources, targets, epochs=20, batch_size=64)
```



### Test

```python
from seq2seq.model import Seq2Seq

model = Seq2Seq('seq2seq-model')
outputs = model.generate('...', beam_size=3, method='greedy')
print(outputs)
```



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

