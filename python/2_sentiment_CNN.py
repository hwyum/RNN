import collections, os, spacy
import pandas as pd
import mxnet as mx
import numpy as np
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.ndarray.linalg import gemm2
from utils import *

os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'


#### Sentence Representation
class Sentence_Representation(nn.Block):
    def __init__(self, **kwargs):
        super(Sentence_Representation, self).__init__()
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        with self.name_scope():
            self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
            self.conv1 = nn.Conv2D(channels = 8, kernel_size = (3, self.emb_dim), activation = 'relu')
            self.maxpool1 = nn.MaxPool2D(pool_size = (self.max_sentence_length -3 + 1, 1), strides = (1, 1))
            self.conv2 = nn.Conv2D(channels = 8, kernel_size = (4, self.emb_dim), activation = 'relu')            
            self.maxpool2 = nn.MaxPool2D(pool_size = (self.max_sentence_length -4 + 1, 1), strides = (1, 1))
            self.conv3 = nn.Conv2D(channels = 8, kernel_size = (5, self.emb_dim), activation = 'relu')
            self.maxpool3 = nn.MaxPool2D(pool_size = (self.max_sentence_length -5 + 1, 1), strides = (1, 1))
            self.conv4 = nn.Conv2D(channels = 8, kernel_size = (6, self.emb_dim), activation = 'relu') 
            self.maxpool4 = nn.MaxPool2D(pool_size = (self.max_sentence_length -6 + 1, 1), strides = (1, 1))

    def forward(self, x):
        embeds = self.embed(x) # batch * time step * embedding
        embeds = embeds.expand_dims(axis = 1)
        _x1 = self.conv1(embeds)
        _x1 = self.maxpool1(_x1)
        _x1 = nd.reshape(_x1, shape = (-1, 8))
        
        _x2 = self.conv2(embeds)
        _x2 = self.maxpool2(_x2)
        _x2 = nd.reshape(_x2, shape = (-1, 8))
        
        _x3 = self.conv3(embeds)
        _x3 = self.maxpool3(_x3)
        _x3 = nd.reshape(_x3, shape = (-1, 8))
        
        _x4 = self.conv4(embeds)
        _x4 = self.maxpool4(_x4)
        _x4 = nd.reshape(_x4, shape = (-1, 8))

        _x = nd.concat(_x1, _x2, _x3, _x4)
        return _x
    
#### Sentiment analysis classifier
class SA_CNN_Classifier(nn.Block):
    def __init__(self, sen_rep, classifier, context, **kwargs):
        super(SA_CNN_Classifier, self).__init__(**kwargs)
        self.context = context
        with self.name_scope():
            self.sen_rep = sen_rep
            self.classifier = classifier
            
    def forward(self, x):
        # sentence representation할 때 hidden의 context가 cpu여서 오류 발생. context를 gpu로 전환
        x = self.sen_rep(x)
        res = self.classifier(x)
        return res




if __name__ == '__main__':
    
    max_sen_len = 20
    max_vocab = 100
    batch_size = 16
    learning_rate = .0002
    log_interval = 100    
    context = mx.cpu()
    
    ### Preprocessing using Spacy
    x, y, origin_txt, idx2word = prepare_data('../data/umich-sentiment-train.txt', max_sen_len, max_vocab)
    pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index()

    ## Data process - tr/va split and define iterator
    tr_idx = np.random.choice(range(len(x)), int(len(x) * .8))
    va_idx = [x for x in range(len(x)) if x not in tr_idx]
    tr_x = [x[i] for i in tr_idx]
    tr_y = [y[i] for i in tr_idx]
    tr_origin = [origin_txt[i] for i in tr_idx]
    va_x = [x[i] for i in va_idx]
    va_y = [y[i] for i in va_idx]
    va_origin = [origin_txt[i] for i in va_idx]
    train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False)
    valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False)

    #### Classifier
    classifier = nn.Sequential()
    classifier.add(nn.Dense(16, activation = 'relu'))
    classifier.add(nn.Dense(8, activation = 'relu'))
    classifier.add(nn.Dense(1))
    classifier.collect_params().initialize(mx.init.Xavier(), ctx = context)

    #### Initiate sentiment classifier
    emb_dim = 50 # Emb dim
    param = {'emb_dim': emb_dim, 'vocab_size': max_vocab, 'max_sentence_length': max_sen_len, 'dropout': .2}
    sen_rep = Sentence_Representation(**param)
    sen_rep.collect_params().initialize(mx.init.Xavier(), ctx = context)

    sa = SA_CNN_Classifier(sen_rep, classifier, context)
    loss = gluon.loss.SigmoidBCELoss()
    trainer = gluon.Trainer(sa.collect_params(), 'adam', {'learning_rate': 1e-3})

    train(5, log_interval, sa, train_data, valid_data, trainer, loss, context = context) 

    ## Classification results
    result = get_pred(sa, loss, valid_data, idx2word, context)
    result[result.pred_sa != result.label].shape
    result[result.pred_sa != result.label].head(10)
    wrong = result[result.pred_sa != result.label]
    for i in range(20):
        print('{} --- Label:{}'.format(wrong['text'].iloc[i], wrong['label'].iloc[i]))