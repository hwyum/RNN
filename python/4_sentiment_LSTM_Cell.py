import os
import pandas as pd
import numpy as np


import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn, rnn
from utils import *
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

class Sentence_Representation(nn.Block): ## Using LSTM
    def __init__(self, emb_dim, hidden_dim, vocab_size, dropout = .2, **kwargs):
        super(Sentence_Representation, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        with self.name_scope():
            self.hidden = []
            self.embed = nn.Embedding(vocab_size, emb_dim)
            self.lstm = rnn.LSTM(hidden_dim // 2, dropout = dropout \
                               , input_size = emb_dim \
                               , bidirectional = True \
                               , layout = 'NTC')

    def forward(self, x, hidden):
        embeds = self.embed(x) # batch * time step * embedding: NTC
        lstm_out, self.hidden = self.lstm(embeds, hidden)
        # (2 * num_layers) * Batch size * Hidden dim 
        return lstm_out, self.hidden

class Sentence_Representation(nn.Block): ## Using LSTMCell : Only use the last time step
    def __init__(self, emb_dim, hidden_dim, vocab_size, dropout = .2, **kwargs):
        super(Sentence_Representation, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        with self.name_scope():
            self.f_hidden = []
            self.b_hidden = []
            self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
            self.drop = nn.Dropout(.2)
            self.f_lstm = rnn.LSTMCell(self.hidden_dim // 2)
            self.b_lstm = rnn.LSTMCell(self.hidden_dim // 2)
            
    def forward(self, x, _f_hidden, _b_hidden):
        embeds = self.embed(x) # batch * time step * embedding
        self.f_hidden = _f_hidden
        self.b_hidden = _b_hidden
        # Forward LSTM
        for i in range(embeds.shape[1]):
            dat = embeds[:, i, :]
            _, self.f_hidden = self.f_lstm(dat, self.f_hidden)
        # Backward LSTM
        for i in np.arange(embeds.shape[1], 0, -1):
            dat = embeds[:, np.int(i - 1), :] # np.int() necessary
            _, self.b_hidden = self.b_lstm(dat, self.b_hidden)
        x = nd.concat(self.f_hidden[0], self.b_hidden[0], dim = 1)
        return x
    
class SA_Classifier(nn.Block):
    def __init__(self, sen_rep, classifier, batch_size, context, **kwargs):
        super(SA_Classifier, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.context = context
        with self.name_scope():
            self.sen_rep = sen_rep
            self.classifier = classifier
            
    def forward(self, x):
        b_hidden = self.sen_rep.b_lstm.begin_state(func = mx.nd.zeros \
                                               , batch_size = self.batch_size \
                                               , ctx = self.context)
        f_hidden = self.sen_rep.f_lstm.begin_state(func = mx.nd.zeros \
                                               , batch_size = self.batch_size \
                                               , ctx = self.context)

        _x = self.sen_rep(x, f_hidden, b_hidden) # Use the last hidden step
        # Extract hidden state from _x
        x = self.classifier(_x)
        return x           
    
    
if __name__ == '__main__':
    # Parameters
    max_sen_len = 20
    max_vocab = 10000
    batch_size = 16
    learning_rate = .0002
    log_interval = 100
    emb_dim = 50 # Emb dim
    hidden_dim = 30 # Hidden dim for LSTM
    context = mx.cpu()
    
    x, y, origin_txt, idx2word = prepare_data('../data/umich-sentiment-train.txt', max_sen_len, max_vocab)

    tr_idx = np.random.choice(range(len(x)), int(len(x) * .8))
    va_idx = [x for x in range(len(x)) if x not in tr_idx]
    tr_x = [x[i] for i in tr_idx]
    tr_y = [y[i] for i in tr_idx]
    va_x = [x[i] for i in va_idx]
    va_y = [y[i] for i in va_idx]
    train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False)
    valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False)

    # Classifier
    classifier = nn.Sequential()
    classifier.add(nn.Dense(16, activation = 'relu'))
    classifier.add(nn.Dense(8, activation = 'relu'))
    classifier.add(nn.Dense(1))
    classifier.collect_params().initialize(mx.init.Xavier(), ctx = context)

    # Sentence representation
    sen_rep = Sentence_Representation(emb_dim, hidden_dim, max_vocab, dropout = .5)
    sen_rep.collect_params().initialize(mx.init.Xavier(), ctx = context)

    # Classifier
    sa = SA_Classifier(sen_rep, classifier,  batch_size, context)
    loss = gluon.loss.SigmoidBCELoss()
    trainer = gluon.Trainer(sa.collect_params(), 'adam', {'learning_rate': 1e-3})

    train(5, log_interval, sa, train_data, valid_data, trainer, loss, context = context) 

    result = get_pred(sa, loss, valid_data, idx2word, context)

    ### Number of wrong classification
    print('wrong classification = {}'.format(len(result[result['pred_sa'] != result['label']])))
          