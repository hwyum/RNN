import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from mxnet import nd, autograd, gluon
import mxnet as mx
from mxnet.gluon import nn, rnn
from utils import *

class calculator(gluon.Block):
    def __init__(self, n_hidden, in_seq_len, out_seq_len, vocab_size, enc_layer, dec_layer = 1, **kwargs):
        super(calculator, self).__init__(**kwargs)
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.enc_layer = enc_layer
        
        with self.name_scope():
            self.encoder = rnn.LSTM(hidden_size = n_hidden, num_layers = enc_layer, layout = 'NTC')
            self.decoder_0 = rnn.LSTMCell(hidden_size = n_hidden)
            self.decoder_1 = rnn.LSTMCell(hidden_size = n_hidden)
            self.batchnorm = nn.BatchNorm(axis = 2)
            self.dense = nn.Dense(self.vocab_size, flatten = False)
            
    def forward(self, inputs, outputs):
        # API says: num_layers, batch_size, num_hidden
        self.batch_size = inputs.shape[0]
        begin_state = self.encoder.begin_state(batch_size = self.batch_size, ctx = ctx)
        enout, (h, c) = self.encoder(inputs, begin_state) # h, c: n_layer * batch_size * n_hidden
        # Pick the hidden states and cell states at the last time step in the second layer
        next_h_0 = h[0] # batch_size * n_hidden
        next_c_0 = c[0] # batch_size * n_hidden
        next_h_1 = h[1] # batch_size * n_hidden
        next_c_1 = c[1] # batch_size * n_hidden
        for i in range(self.out_seq_len):
            deout, (next_h_0, next_c_0) = self.decoder_0(outputs[:, i, :], [next_h_0, next_c_0],)
            deout, (next_h_1, next_c_1) = self.decoder_1(deout, [next_h_1, next_c_1],)
            if i == 0:
                deouts = deout
            else:
                deouts = nd.concat(deouts, deout, dim = 1)   
        deouts = nd.reshape(deouts, (-1, self.out_seq_len, self.n_hidden))
        deouts = self.batchnorm(deouts)
        deouts_fc = self.dense(deouts)
        return deouts_fc
    
    def calculation(self, input_str, char_indices, indices_char, input_digits = 9, lchars = 14, ctx = mx.cpu()):
        input_str = 'S' + input_str + 'E'
        X = nd.zeros((1, input_digits, lchars), ctx = ctx)
        for t, char in enumerate(input_str):
            X[0, t, char_indices[char]] = 1
        Y_init = nd.zeros((1, lchars), ctx = ctx)
        Y_init[0, char_indices['S']] = 1
        begin_state = self.encoder.begin_state(batch_size = 1, ctx = ctx)
        enout, (h, c) = self.encoder(X, begin_state)
        next_h_0 = h[0] # batch_size * n_hidden
        next_c_0 = c[0] # batch_size * n_hidden
        next_h_1 = h[1] # batch_size * n_hidden
        next_c_1 = c[1] # batch_size * n_hidden
        deout = Y_init
        
        for i in range(self.out_seq_len):
            deout, (next_h_0, next_c_0) = self.decoder_0(deout, [next_h_0, next_c_0],)
            deout, (next_h_1, next_c_1) = self.decoder_1(deout, [next_h_1, next_c_1],)
            deout = nd.expand_dims(deout, axis = 1)
            deout = self.batchnorm(deout)
            deout = deout[:, 0, :]
            deout_sm = self.dense(deout)
            deout = nd.one_hot(nd.argmax(nd.softmax(deout_sm, axis = 1), axis = 1), depth = self.vocab_size)
            if i == 0:
                ret_seq = indices_char[nd.argmax(deout_sm, axis = 1).asnumpy()[0].astype('int')]
            else:
                ret_seq += indices_char[nd.argmax(deout_sm, axis = 1).asnumpy()[0].astype('int')]

            if ret_seq[-1] == ' ' or ret_seq[-1] == 'E':
                break
        return ret_seq.strip('E').strip()
 

if __name__ == '__main__':
    
    ctx = mx.cpu()
    
    # Obs.
    N = 50000
    N_train = int(N * .9)
    N_validation = N - N_train
    
    # Number of digits
    digits = 3
    input_digits = digits * 2 + 3
    output_digits = digits + 3

    added = set()
    questions = []
    answers = []
    answers_y = []

    # Simulate data
    while len(questions) < N:
        a, b = n(), n()
        pair = tuple(sorted((a, b)))
        if pair in added:
            continue
        question = 'S{}+{}E'.format(a, b)
        question = padding(question, input_digits)
        answer = 'S' + str(a + b) + 'E'
        answer = padding(answer, output_digits)
        answer_y = str(a + b) + 'E'
        answer_y = padding(answer_y, output_digits)
        added.add(pair)
        questions.append(question)
        answers.append(answer)
        answers_y.append(answer_y)
    
    # Make corpus
    chars = '0123456789+SE '
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # Create data
    X = np.zeros((len(questions), input_digits, len(chars)), dtype=np.integer)
    Y = np.zeros((len(questions), digits + 3, len(chars)), dtype=np.integer)
    Z = np.zeros((len(questions), digits + 3, len(chars)), dtype=np.integer)

    for i in range(N):
        for t, char in enumerate(questions[i]):
            X[i, t, char_indices[char]] = 1
        for t, char in enumerate(answers[i]):
            Y[i, t, char_indices[char]] = 1
        for t, char in enumerate(answers_y[i]):
            Z[i, t, char_indices[char]] = 1

    X_train, X_validation, Y_train, Y_validation, Z_train, Z_validation = \
        train_test_split(X, Y, Z, train_size=N_train)
        
    # Data loader
    tr_set = gluon.data.ArrayDataset(X_train, Y_train, Z_train)
    tr_data_iterator = gluon.data.DataLoader(tr_set, batch_size=256, shuffle=True)
    va_set =gluon.data.ArrayDataset(X_validation, Y_validation, Z_validation)
    va_data_iterator = gluon.data.DataLoader(va_set, batch_size=256, shuffle=True)

    # Define model
    model = calculator(300, 9, 6, 14, 2)
    model.collect_params().initialize(mx.init.Xavier(), ctx = ctx)
    model.hybridize()
    trainer = gluon.Trainer(model.collect_params(), 'rmsprop')
    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis = 2, sparse_label = False)

    # Train model
    _, _  = train_seq2seq(2, 10, model, tr_data_iterator, va_data_iterator, trainer, loss, char_indices, indices_char, ctx) 

    

