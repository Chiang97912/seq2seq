# -*- coding: utf-8 -*-
from .dataset import Dataset
from .tokenizer import Tokenizer
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torchtext.data import BucketIterator
from torch.optim import lr_scheduler
import json
import math
import os
import random
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


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


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, dropout=(0 if num_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        embedded = self.embed(input)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(0))
        output = F.log_softmax(output, dim=1)
        attn_weights = torch.zeros([encoder_outputs.size(0), 1, encoder_outputs.size(2)])
        return output, hidden, attn_weights


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def score(self, hidden, encoder_outputs):
        # [B,T,2H]->[B,T,H]
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy.squeeze(1)  # [B,T]

    def forward(self, hidden, encoder_outputs, mask=None):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(h, encoder_outputs)
        if mask is not None:
            attn_energies.data.masked_fill_(mask, -float('inf'))
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 num_layers=1, dropout=0.2):
        super(BahdanauAttentionDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          num_layers, dropout=(0 if num_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs, mask=None):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs, mask)
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


class LuongAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super(LuongAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def score(self, hidden, encoder_outputs):
        # hidden: [batch_size,1,hidden_size]; encoder_outputs: [batch_size,seq_len,hidden_size]
        if self.method == 'dot':
            return torch.sum(hidden * encoder_outputs, dim=2)  # element-wise multiply
        elif self.method == 'general':
            energy = self.attn(encoder_outputs)  # [batch_size,seq_len,hidden_size]
            return torch.sum(hidden * energy, dim=2)
        elif self.method == 'concat':
            energy = torch.tanh(self.attn(torch.cat((hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), dim=2)))  # [batch_size,seq_len,hidden_size]
            return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, mask=None):
        attn_energies = self.score(hidden, encoder_outputs)  # [batch_size,seq_len]
        if mask is not None:
            attn_energies.data.masked_fill_(mask, -float('inf'))
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch_size,1,seq_len]


class LuongAttentionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, num_layers=1, dropout=0.2, attention_method='dot'):
        super(LuongAttentionDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, dropout=(0 if num_layers == 1 else dropout))
        self.attention = LuongAttention(attention_method, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs, mask=None):
        # input: [1,batch_size]; last_hidden: [num_layers,batch_size,hidden_size]; encoder_outputs: [seq_len,batch_size,hidden_size]
        # Get embedding of current input word
        embedded = self.embed(input)  # [1,batch_size,embed_size]
        embedded = self.dropout(embedded)

        output, hidden = self.gru(embedded, last_hidden)  # output: [1,batch_size,hidden_size]; hidden: [num_layers,batch_size,hidden_size]

        output = output.transpose(0, 1)  # [batch_size,1,hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size,seq_len,hidden_size]
        # Calculate attention weights from the current GRU output
        attn_weights = self.attention(output, encoder_outputs, mask)  # [batch_size,1,seq_len]
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size,1,hidden_size]

        output = output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((output, context), dim=1)
        concat_output = torch.tanh(self.concat(concat_input))  # [batch_size,hidden_size]
        output = self.out(concat_output)
        output = F.log_softmax(output, dim=1)  # [batch_size,output_size]
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
                 decoder_dropout=0.5,
                 attention='luong',
                 attention_method='dot'):
        """Sequence to Sequence Model with Attention

        Industrial-grade implementation of seq2seq algorithm based on Pytorch, integrated beam search algorithm.

        Keyword Arguments:
            model_name_or_path {str} -- The name or path of model (default: {None})
            encoder {torch.nn.Module} -- Encoder model (default: {None})
            decoder {torch.nn.Module} -- Decoder model (default: {None})
            device {str} -- Device (like 'cuda' / 'cpu') that should be used for computation (default: {'cpu'})
            embed_size {number} -- Embedding size (default: {256})
            hidden_size {number} -- Hidden size (default: {512})
            max_seq_length {number} -- Maximum sequence length (default: {64})
            encoder_num_layers {number} -- The number of encoder GRU layers (default: {2})
            decoder_num_layers {number} -- The number of decoder GRU layers (default: {1})
            lang4src {str} -- Source text language {zh, en} (default: {None})
            lang4tgt {str} -- Target text language {zh, en} (default: {None})
            encoder_dropout {number} -- Dropout rate for encoder (default: {0.5})
            decoder_dropout {number} -- Dropout rate for decoder (default: {0.5})
            attention {str} -- Attention name {bahdanau, luong, None} (default: {'luong'})
            attention_method {str} -- Luong attention method, only works when attention="luong" {dot, general, concat} (default: {'dot'})
        """
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
        self.attention = attention
        self.attention_method = attention_method
        self.config_keys = ['input_size', 'embed_size', 'hidden_size', 'max_seq_length', 'encoder_num_layers',
                            'decoder_num_layers', 'output_size', 'lang4src', 'lang4tgt', 'attention', 'attention_method']

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
            if self.attention == 'bahdanau':
                self.decoder = BahdanauAttentionDecoder(self.embed_size, self.hidden_size, self.output_size, num_layers=self.decoder_num_layers, dropout=self.decoder_dropout)
            elif self.attention == 'luong':
                self.decoder = LuongAttentionDecoder(self.embed_size, self.hidden_size, self.output_size, num_layers=self.decoder_num_layers, dropout=self.decoder_dropout, attention_method=self.attention_method)
            elif self.attention is None:
                self.decoder = Decoder(self.embed_size, self.hidden_size, self.output_size, num_layers=self.decoder_num_layers, dropout=self.decoder_dropout)
            else:
                raise ValueError(self.attention, "is not an appropriate attention type.")
        else:
            self.decoder = decoder
        self.to(self.device)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = tgt.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).to(self.device)

        encoder_output, hidden = self.encoder(src)
        mask = (src == self.tokenizer4src.pad_id).transpose(1, 0)
        hidden = hidden[:self.decoder.num_layers]
        output = Variable(tgt.data[0, :]).unsqueeze(0)  # sos token
        outputs[0] = self.tokenizer4tgt.sos_id
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output, mask=mask)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(tgt.data[t] if is_teacher else top1).to(self.device)
            output = output.unsqueeze(0)
        return outputs

    def fit(self, sources, targets, epochs=20, lr=0.001, batch_size=16, train_size=0.8, grad_clip=10,
            weight_decay=0, lr_decay=1, teacher_forcing_ratio=0.5, flood_level=0, print_every=100):
        """

        Train the model with source and target pairs.

        Arguments:
            sources {list} -- source text list
            targets {list} -- target text list

        Keyword Arguments:
            epochs {number} -- Number of epochs for training (default: {20})
            lr {number} -- Learning rate (default: {0.001})
            batch_size {number} -- Batch size (default: {16})
            train_size {number} -- Training dataset ratio (default: {0.8})
            grad_clip {number} -- Max norm of the gradients (default: {10})
            weight_decay {number} -- Weight decay for optimizer(L2 Regularization) (default: {0})
            lr_decay {number} -- Multiplicative factor of learning rate decay (default: {1})
            teacher_forcing_ratio {number} -- Teacher forcing ratio (default: {0.5})
            flood_level {number} -- Flood level for loss (default: {0})
            print_every {number} -- The number of steps to print (default: {100})
        """
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

        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': weight_decay},
                                      {'params': bias_p, 'weight_decay': 0}], lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=lr_decay)
        pad = self.tokenizer4tgt.pad_id
        vocab_size = len(self.tokenizer4tgt.vocab)

        best_val_loss = None
        for e in range(1, epochs+1):
            self.train()
            scheduler.step()
            total_loss = 0
            total_score = 0
            total_batch = len(train_iter)
            for b, batch in enumerate(train_iter):
                src = Variable(batch.source).to(self.device)
                tgt = Variable(batch.target).to(self.device)
                optimizer.zero_grad()
                output = self.forward(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                  tgt[1:].contiguous().view(-1),
                                  ignore_index=pad)
                refs = tgt[1:].permute(1, 0).unsqueeze(1).tolist()
                pred = output[1:].permute(1, 0, 2).data.max(2)[1].tolist()
                score = corpus_bleu(refs, pred, weights=(1, 0, 0, 0))
                total_score += score
                # loss.backward()
                flood = (loss - flood_level).abs() + flood_level
                flood.backward()
                clip_grad_norm_(self.parameters(), grad_clip)  # fix exploding gradients problem
                optimizer.step()
                total_loss += loss.data.item()

                if b % print_every == 0 and b != 0:
                    total_loss = total_loss / print_every
                    total_score = total_score / print_every
                    print("[%d/%s][loss:%5.2f][pp:%5.2f][bleu:%5.2f]" %
                          (b, total_batch, total_loss, math.exp(total_loss), total_score))
                    total_loss = 0

            with torch.no_grad():
                self.eval()
                val_loss = 0
                val_score = 0
                for b, batch in enumerate(val_iter):
                    src = Variable(batch.source).to(self.device)
                    tgt = Variable(batch.target).to(self.device)
                    output = self.forward(src, tgt, teacher_forcing_ratio=0.0)
                    loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                      tgt[1:].contiguous().view(-1),
                                      ignore_index=pad)
                    val_loss += loss.data.item()
                    refs = tgt.permute(1, 0).unsqueeze(1).tolist()
                    pred = output.permute(1, 0, 2).data.max(2)[1].tolist()
                    score = corpus_bleu(refs, pred, weights=(1, 0, 0, 0))
                    val_score += score
                val_loss = val_loss / len(val_iter)
                val_score = val_score / len(val_iter)

            print("[Epoch:%d/%s] val_loss:%5.3f | val_pp:%5.2fS | val_bleu: %5.2f"
                  % (e, epochs, val_loss, math.exp(val_loss), val_score))

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            print("[!] saving model...")
            if not os.path.isdir(self.model_name_or_path):
                os.makedirs(self.model_name_or_path)
            torch.save(self.state_dict(), self.model_path)

    def generate(self, src, method='beam', sep=None, max_len=64, beam_size=3, diversity_rate=1, sample_bias=-1):
        """

        Generate target from source text.

        Arguments:
            src {str} -- source text

        Keyword Arguments:
            method {str} -- Search algorithm name {greedy, beam} (default: {'beam'})
            sep {str} -- String inserted between outputs (default: {None})
            max_len {number} -- Maximum sequence length (default: {64})
            beam_size {number} -- Beam size a parameter in the beam search algorithm which determines
                                  how many of the best partial solutions to evaluate (default: {3})
            diversity_rate {number} -- Diversity rate for beam search algorithm (default: {1})
            sample_bias {number} -- The bias from best result, whilch is smaller than beam_size (default: {-1})

        Returns:
            list -- Target tokens
        """
        token_ids = self.tokenizer4src.encode(src)
        src = Variable(torch.LongTensor(token_ids))
        if method == 'greedy':
            outputs = self.greedy_search(src, max_len=max_len)
        elif method == 'beam':
            outputs = self.beam_search(src, max_len=max_len, beam_size=beam_size, diversity_rate=diversity_rate, sample_bias=sample_bias)
        else:
            raise Exception('method is invalid')
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

    def beam_search(self, src, max_len=64, beam_size=3, diversity_rate=1, sample_bias=-1):
        eos_id = self.tokenizer4tgt.eos_id
        sos_id = self.tokenizer4tgt.sos_id

        with torch.no_grad():
            self.eval()
            src = src.repeat(1, beam_size)
            encoder_output, hidden = self.encoder(src)
            hidden = hidden[:self.decoder.num_layers]
            output = Variable(torch.LongTensor([[sos_id] * beam_size]))  # sos
            outputs = Variable(torch.LongTensor([[sos_id] * beam_size]))
            scores = torch.zeros(1, 1)
            for t in range(1, max_len):
                output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
                if t == 1:
                    output = output[0].unsqueeze(0)

                _, indices = output.data.topk(output.shape[-1])
                ranks = torch.arange(1, output.shape[-1] + 1).repeat(output.shape[0], 1) * diversity_rate
                ranks.scatter_(1, indices, ranks.clone())
                output = output - ranks

                new_scores = scores.view(-1, 1) + output  # accumulative scores
                new_scores = new_scores.view(-1)  # flatten
                topk_scores, topk_indices = new_scores.data.topk(beam_size)
                rows = (topk_indices // output.shape[-1])  # row indices
                columns = (topk_indices % output.shape[-1])  # column indices

                scores = topk_scores  # update scores
                output = columns.unsqueeze(0).long()  # update decoder input

                best_index = scores.argmax(dim=0)
                outputs = torch.cat([outputs[:, rows], output], dim=0)  # record sequence
            if sample_bias < 0:
                sample_bias = beam_size + sample_bias
            index = random.randint(0, min(beam_size - 1, sample_bias))
            return outputs[1:, index]

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
