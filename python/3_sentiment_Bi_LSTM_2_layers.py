import os, re, tqdm
import pandas as pd
import mxnet as mx
import numpy as np

from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from utils import *

os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'


class Sentence_Representation(nn.Block):
    def __init__(self, emb_dim, hidden_dim, vocab_size, dropout = .2, **kwargs):
        super(Sentence_Representation, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        with self.name_scope():
            self.hidden = []
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.lstm = rnn.LSTM(hidden_dim // 2, num_layers= 2, dropout = dropout, input_size = emb_dim, bidirectional=True)
            self.drop = nn.Dropout(.2)

    def forward(self, x, hidden):
        embeds = self.embed(x) # batch * time step * embedding: NTC
        lstm_out, self.hidden = self.lstm(nd.transpose(embeds, (1, 0, 2)), hidden) #TNC로 변환
        _hid = [nd.transpose(x, (1, 0, 2)) for x in self.hidden]
        # Concatenate depreciated. use concat. input list of tensors
        _hidden = nd.concat(*_hid)
        return lstm_out, self.hidden

    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)
    
    
class SA_Classifier(nn.Block):
    def __init__(self, sen_rep, classifier, batch_size, context, **kwargs):
        super(SA_Classifier, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.context = context
        with self.name_scope():
            self.sen_rep = sen_rep
            self.classifier = classifier
            
    def forward(self, x):
        hidden = self.sen_rep.begin_state(func = mx.nd.zeros, batch_size = self.batch_size, ctx = self.context)
        #_x, _ = self.sen_rep(x, hidden)
        # Use the last cell state from both directions
        _, _x = self.sen_rep(x, hidden) 
        print('x shape = {}'.format(_x[0].shape)) # Hidden state
        print('x shape = {}'.format(_x[1].shape)) # Cell state
        # state = (2 * num_layers, batch_size, num_hidden): 2 for left and right LSTM
        # Select the last layer for both of left LSTM and right LSTM
        x = nd.concat(_x[1][1, :, :], _x[1][3, :, :], dim = -1)
        x = self.classifier(x)
        return x  

if __name__ == '__main__':
    max_sen_len = 5
    max_vocab = 100
    batch_size = 16
    learning_rate = .0002
    log_interval = 100
    emb_dim = 50 # Emb dim
    hidden_dim = 30 # Hidden dim for LSTM
    context = mx.cpu()
    
    x, y, origin_txt, idx2word = prepare_data('../data/umich-sentiment-train.txt', max_sen_len, max_vocab)
        
    ## Data process - tr/va split and define iterator
    tr_idx = np.random.choice(range(len(x)), int(len(x) * .8))
    va_idx = [x for x in range(len(x)) if x not in tr_idx]
    tr_x = [x[i] for i in tr_idx]
    tr_y = [y[i] for i in tr_idx]
    va_x = [x[i] for i in va_idx]
    va_y = [y[i] for i in va_idx]
    train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False)
    valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False)
    
    # sentence representation
    sen_rep = Sentence_Representation(emb_dim, hidden_dim, max_vocab)
    
    # classifier
    classifier = nn.Sequential()
    classifier.add(nn.Dense(16, activation = 'relu'))
    classifier.add(nn.Dense(8, activation = 'relu'))
    classifier.add(nn.Dense(1))

    # Sentiment analysis class
    sa = SA_Classifier(sen_rep, classifier,  batch_size, context)
    sa.collect_params().initialize(mx.init.Xavier(), ctx = context)
    loss = gluon.loss.SigmoidBCELoss()
    trainer = gluon.Trainer(sa.collect_params(), 'adam', {'learning_rate': 1e-3})

    # Train model
    train(5, log_interval, sa, train_data, valid_data, trainer, loss, context = context) 
    
    # We need to specify batch_size explicitly becuase we need that in reshaping
    idx = np.random.choice(len(va_idx), batch_size)
    va_txt = [origin_txt[_idx] for _idx in va_idx]
    va_txt = [va_txt[j] for j in idx]
    va_txt = pd.DataFrame(va_txt, columns = ['txt'])
    y_pred_sa = sa(nd.array([va_x[i] for i in idx], ctx = context))
    pred_sa = [nd.round(val).asnumpy() for val in nd.sigmoid(y_pred_sa)] 
    pred_sa_pd = pd.DataFrame(pred_sa, columns  = ['pred_sa'])
    label_pd = pd.DataFrame([va_y[j] for j in idx], columns = ['label'])
    result = pd.concat([va_txt, pred_sa_pd, label_pd], axis = 1)
    result.head(10)

    result[result['pred_sa'] != result['label']].shape