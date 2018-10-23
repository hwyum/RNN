import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from mxnet import nd, autograd, gluon
import mxnet as mx
from mxnet.gluon import nn, rnn
from utils import *

class calculator(gluon.Block):
    def __init__(self, n_hidden, in_seq_len, out_seq_len, vocab_size, **kwargs):
        super(calculator, self).__init__(**kwargs)
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        
        with self.name_scope():
            self.encoder = rnn.LSTMCell(hidden_size = n_hidden)
            self.decoder = rnn.LSTMCell(hidden_size = n_hidden)
            self.batchnorm = nn.BatchNorm(axis = 2)
            self.dense = nn.Dense(self.vocab_size, flatten = False)
            
    def forward(self, inputs, outputs):
        # Since we don't use intermediate states for 'though vector', we don't need to unroll it.
        # In the later examples, we will use LSTM class rather than LSTMCell class.
        enout, (next_h, next_c) = self.encoder.unroll(inputs = inputs
                                                    , length = self.in_seq_len
                                                    , merge_outputs = True)
        for i in range(self.out_seq_len):
            deout, (next_h, next_c) = self.decoder(outputs[:, i, :], [next_h, next_c],)
            if i == 0:
                deouts = deout
            else:
                deouts = nd.concat(deouts, deout, dim = 1)
            #print('i= {}, deouts= {}'.format(i, deouts.shape))
        
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
        enout, (next_h, next_c) = self.encoder.unroll(inputs = X, length = self.in_seq_len, merge_outputs = True)
        deout = Y_init
        
        for i in range(self.out_seq_len):
            deout, (next_h, next_c) = self.decoder(deout, [next_h, next_c])
            #print('dim deout = {}'.format(deout.shape))
            deout = nd.expand_dims(deout, axis = 1)
            #print('dim deout = {}'.format(deout.shape))
            deout = self.batchnorm(deout)
            deout = deout[:, 0, :]
            #print('dim deout = {}'.format(deout.shape))

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
    model = calculator(300, 9, 6, 14)
    model.collect_params().initialize(mx.init.Xavier(), ctx = ctx)
    trainer = gluon.Trainer(model.collect_params(), 'rmsprop')
    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis = 2, sparse_label = False)

    # Train model
    _, _  = train_seq2seq(5, 10, model, tr_data_iterator, va_data_iterator, trainer, loss, char_indices, indices_char, ctx) 

    

