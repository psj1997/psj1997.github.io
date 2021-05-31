---
layout: post
title: Notes of Sequence to Sequence Learning with Neural Networks
category: deep learning
tags: [seq2seq， deep learning]
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>


# Notes of Sequence to Sequence Learning with Neural Networks #

This blog is a note of the paper `Sequence to Sequence Learning with Neural Networks`, which used sequence-to-sequence model on NMT(Neural Machine Translation) task. In this blog, I will describe the idea of the seq2seq model and  a simple code implementation with python and pytorch.

In this paper, the authors proposed a end-to-end seq2seq model on language translation task. The basic idea of the model is the model contains a encoder and a decoder. The encoder and decoder uses RNN(LSTM, GRU). The encoder will map the input sequences to a fixed dimensionality vector and the decoder will decode the vector to get the target sequence. In this paper, the authors used a multi-layer LSTM to encode and decode the  sequences. The basic structure is in the following picture(from original paper). The advantage of LSTM used here is that it can process variable-length sequence and it can learn the long-term dependence information.

![model](/assets/img/blog1_1.jpg)

As shown in the picture, the encoder reads the input sequence 'ABC' and map them to a fixed embedding feature. Then the decoder reveives the feature(hidden state) from encoder and outputs the 'WXYZ'.

Here are the descriptions of the encoder and decoder in the paper: 

Given a sequence of inputs ($x_1, x_2, ..., x_T$),  the goal of the model is to estimate the conditional probability $p(y_1, y_2, ..., y_{T'}|x_1, ..., x_T)$.  ($y_1, y_2, ..., y_{T'}$) is the corresponding output sequences. 

$$ p(y_1, y_2, ..., y_{T'}|x_1, ..., x_T) = \prod_{t=1}^{T'}p(y_t|v, y_1, ..., y_{t-1}) $$

$v$ is the last hidden state from the LSTM.

Compared to previous work, the models here have some differences: (1) two different LSTMs for encoder and decoder; (2) deep LSTMs; (3) reverse the order of the words of the input sentences. The authors estimated that eversing could introduce many short term dependencies to the dataset. This way would lead to better predictions in the early parts of the target sequences.

**Beam search algorithm** :

 In the prediction step, we need to calculate the conditional probability of the output sequences. It takes much time if you calculate all situations and get the best. One idea to deal with it is greedy search. Every time when we generate the word, we select the best and get the final output sequences. The question of this is that it sometimes can not get very good result. One way used in the paper and other NMT task is beam search. it has a very similar idea of greedy search with a beam size(*k*) parameter. The difference of it compared to greedy search is that it selects *k* best word every time. So *k* = 1 is just greedy search.

Here are peseude codes of this model. There includes 3 part: encoder, decoder and seq2seq model.

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hiddien_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        ### you can use RNN/LSTM/GRU here
        self.rnn = nn.RNN(input_dim, hiddien_dim, n_layers, dropout)
        
    def forward(self, src):
        embedded = self.embedding(src) ### src: source sequences
        ### In machine translation task, we used the last hidden state of evert rnn layer 
        outputs, hidden = self.rnn(embedded)
        return hidden
```

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hiddien_dim, n_layers, dropout):
		super().__init__()
                self.embedding = nn.Embedding(input_dim, embedding_dim)
        ### you can use RNN/LSTM/GRU here
        self.rnn = nn.RNN(input_dim, hiddien_dim, n_layers, dropout)
        self.linear = n.Linear(hid_dim, output_dim)
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.linear(output)
        return prediction, hidden
```

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        outputs = torch.zeros(target_length, batch_size, output_dim)
        hidden = self.encoder(src)
        input = trg[0,:] ### the start symbol
        for t in range(1, target_length):
            # every iteration generating a word according to input word and hidden state 
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1
        return outputs
```



