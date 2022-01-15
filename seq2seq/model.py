# -*- coding: utf-8 -*-
from .dataset import Dataset
from .tokenizer import Tokenizer
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torchtext.data import BucketIterator
import json
import math
import os
import random
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 num_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B,T,2H]->[B,T,H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy.squeeze(1)  # [B,T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 encoder=None,
                 decoder=None,
                 device='cpu',
                 embed_size=256,
                 hidden_size=512,
                 max_seq_length=64,
                 encoder_num_layers=2,
                 decoder_num_layers=1,
                 lang4src=None,
                 lang4tgt=None,
                 encoder_dropout=0.5,
                 decoder_dropout=0.5):
        super(Seq2Seq, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.lang4src = lang4src
        self.lang4tgt = lang4tgt
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.config_keys = ['input_size', 'embed_size', 'hidden_size', 'max_seq_length', 'encoder_num_layers',
                            'decoder_num_layers', 'output_size', 'lang4src', 'lang4tgt']

        self.model_path = os.path.join(self.model_name_or_path, 'pytorch_model.bin')
        self.vocabulary_path = os.path.join(self.model_name_or_path, 'vocabulary.bin')
        self.vocabulary2_path = os.path.join(self.model_name_or_path, 'vocabulary2.bin')
        self.config_path = os.path.join(self.model_name_or_path, 'config.json')

        self.load()

    def build(self, encoder=None, decoder=None):
        if encoder is None:
            self.encoder = Encoder(self.input_size, self.embed_size, self.hidden_size, num_layers=self.encoder_num_layers, dropout=self.encoder_dropout)
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = Decoder(self.embed_size, self.hidden_size, self.output_size, num_layers=self.decoder_num_layers, dropout=self.decoder_dropout)
        else:
            self.decoder = decoder
        self.to(self.device)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = tgt.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).to(self.device)

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.num_layers]
        output = Variable(tgt.data[0, :]).unsqueeze(0)  # sos token
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(tgt.data[t] if is_teacher else top1).to(self.device)
            output = output.unsqueeze(0)
        return outputs

    def fit(self, sources, targets, epochs=20, lr=0.001, batch_size=16, train_size=0.8, grad_clip=10):
        dataset = Dataset(sources, targets, self.tokenizer4src, self.tokenizer4tgt)
        self.tokenizer4src.build_vocab(dataset)
        if self.lang4tgt != self.lang4src:
            self.tokenizer4tgt.build_vocab(dataset)
        self.save()
        self.build()

        train_dataset, val_dataset = dataset.split(split_ratio=train_size)

        train_iter, val_iter = BucketIterator.splits(
            (train_dataset, val_dataset),
            batch_sizes=(batch_size, batch_size),
            device=0,
            sort_key=lambda x: len(x.source),
            sort_within_batch=False,
            repeat=False
        )

        optimizer = optim.Adam(self.parameters(), lr=lr)
        pad = self.tokenizer4tgt.pad_id
        vocab_size = len(self.tokenizer4tgt.vocab)

        best_val_loss = None
        for e in range(1, epochs+1):
            self.train()
            total_loss = 0
            for b, batch in enumerate(train_iter):
                src = Variable(batch.source).to(self.device)
                tgt = Variable(batch.target).to(self.device)
                optimizer.zero_grad()
                output = self.forward(src, tgt)
                loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                  tgt[1:].contiguous().view(-1),
                                  ignore_index=pad)
                loss.backward()
                clip_grad_norm_(self.parameters(), grad_clip)  # prevent gradient explosion
                optimizer.step()
                total_loss += loss.data.item()

                if b % 100 == 0 and b != 0:
                    total_loss = total_loss / 100
                    print("[%d][loss:%5.2f][pp:%5.2f]" %
                          (b, total_loss, math.exp(total_loss)))
                    total_loss = 0

            with torch.no_grad():
                self.eval()
                val_loss = 0
                for b, batch in enumerate(val_iter):
                    src = Variable(batch.source).to(self.device)
                    tgt = Variable(batch.target).to(self.device)
                    output = self.forward(src, tgt, teacher_forcing_ratio=0.0)
                    loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                      tgt[1:].contiguous().view(-1),
                                      ignore_index=pad)
                    val_loss += loss.data.item()
                val_loss = val_loss / len(val_iter)

            print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
                  % (e, val_loss, math.exp(val_loss)))

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                print("[!] saving model...")
                if not os.path.isdir(self.model_name_or_path):
                    os.makedirs(self.model_name_or_path)
                torch.save(self.state_dict(), self.model_path)
                best_val_loss = val_loss

    def generate(self, src, method='beam', sep=None, beam_size=3):
        token_ids = self.tokenizer4src.encode(src)
        src = Variable(torch.LongTensor(token_ids))
        if method == 'greedy':
            outputs = self.greedy_search(src)
        elif method == 'beam':
            outputs = self.beam_search(src, beam_size=beam_size)
        outputs = self.tokenizer4tgt.decode(outputs, sep=sep)
        return outputs

    def greedy_search(self, src, max_len=64):
        eos_id = self.tokenizer4tgt.eos_id
        sos_id = self.tokenizer4tgt.sos_id

        with torch.no_grad():
            outputs = []
            self.eval()
            encoder_output, hidden = self.encoder(src)
            hidden = hidden[:self.decoder.num_layers]
            output = Variable(torch.LongTensor([[sos_id]]))  # sos
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(
                        output, hidden, encoder_output)
                output = output.long()
                top1 = output.data.max(1)[1]
                output = top1.unsqueeze(0)
                if output[0] == eos_id:
                    break
                outputs.append(output)
            return outputs

    def beam_search(self, src, max_len=64, beam_size=3):
        eos_id = self.tokenizer4tgt.eos_id
        sos_id = self.tokenizer4tgt.sos_id

        with torch.no_grad():
            self.eval()
            src = src.repeat(1, beam_size)
            encoder_output, hidden = self.encoder(src)
            hidden = hidden[:self.decoder.num_layers]
            output = Variable(torch.LongTensor([[sos_id] * beam_size]))  # sos
            scores = torch.zeros(beam_size)
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(
                        output, hidden, encoder_output)

                new_scores = scores.view(-1, 1) + output  # accumulative scores
                new_scores = new_scores.view(-1)  # flatten
                topk_scores, topk_indices = new_scores.data.topk(beam_size)
                rows = (topk_indices // output.shape[-1])  # row indices
                columns = (topk_indices % output.shape[-1])  # column indices

                scores = topk_scores  # update scores
                output = columns.unsqueeze(0).long()  # update decoder input

                eos_cnts = (output == eos_id).sum(0)
                best_index = scores.argmax(dim=0)
                if eos_cnts[best_index] == 1:
                    break
                if t == 1:
                    outputs = output
                else:
                    outputs = torch.cat([outputs[:, rows], output], dim=0)  # record sequence
            return outputs[:, best_index]

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def _save_config(self):
        config = self.get_config_dict()
        os.makedirs(self.model_name_or_path, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def _load_config(self):
        os.makedirs(self.model_name_or_path, exist_ok=True)
        with open(self.config_path, encoding='utf-8') as f:
            config = json.load(f)
        self.__dict__.update(config)

    def _save_vocab(self):
        os.makedirs(self.model_name_or_path, exist_ok=True)
        self.tokenizer4src.save(self.vocabulary_path)
        if self.lang4tgt != self.lang4src:
            self.tokenizer4tgt.save(self.vocabulary2_path)
        self.input_size = len(self.tokenizer4src.vocab)
        self.output_size = len(self.tokenizer4tgt.vocab)

    def _load_vocab(self):
        self.tokenizer4src.load(self.vocabulary_path)
        if self.lang4tgt != self.lang4src:
            self.tokenizer4tgt.load(self.vocabulary2_path)
        self.input_size = len(self.tokenizer4src.vocab)
        self.output_size = len(self.tokenizer4tgt.vocab)

    def save(self):
        self._save_vocab()
        self._save_config()

    def load(self):
        if os.path.exists(self.config_path):
            self._load_config()

        if self.lang4src is not None and self.lang4tgt is not None:
            if self.lang4src != self.lang4tgt:
                self.tokenizer4src = Tokenizer(lang=self.lang4src, fix_length=self.max_seq_length)
                self.tokenizer4tgt = Tokenizer(lang=self.lang4tgt, fix_length=self.max_seq_length)
            else:
                self.tokenizer4src = self.tokenizer4tgt = Tokenizer(lang=self.lang4src, fix_length=self.max_seq_length)

        if os.path.exists(self.vocabulary_path):
            self._load_vocab()
            self.build()

        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path, map_location=self.device))
