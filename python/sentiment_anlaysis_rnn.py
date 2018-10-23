import collections, nltk, os, time
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from sklearn.metrics import accuracy_score, auc
from sklearn.preprocessing import normalize
from tqdm import tqdm, tqdm_notebook



MAX_SENTENCE_LENGTH = 20
MAX_VOCAB = 10000


class Sentence_Representation(nn.Block):
    def __init__(self, emb_dim, hidden_dim, vocab_size, dropout = .2, **kwargs):
        super(Sentence_Representation, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        with self.name_scope():
            self.hidden = []
            self.embed = nn.Embedding(vocab_size, emb_dim)
            self.lstm = rnn.LSTM(hidden_dim // 2, num_layers= 2, dropout = dropout, input_size = emb_dim, bidirectional=True)
            self.drop = nn.Dropout(.2)

    def forward(self, x, hidden):
        #print('x = {}'.format(x))
        embeds = self.embed(x) # batch * time step * embedding: NTC
        lstm_out, self.hidden = self.lstm(nd.transpose(embeds, (1, 0, 2)), hidden) #TNC로 변환
        _hid = [nd.transpose(x, (1, 0, 2)) for x in self.hidden]
        print('_hid len = {}'.format(len(_hid)))
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
        hidden = self.sen_rep.begin_state(func = mx.nd.zeros
                                        , batch_size = self.batch_size
                                        , ctx = self.context)
        print('hidden shape = {}'.format([x.shape for x in hidden]))
        #_x, _ = self.sen_rep(x, hidden)
        _, _x = self.sen_rep(x, hidden) # Use the last hidden step
        print('x shape = {}'.format(_x[0].shape))
        x = nd.reshape(x, (-1,))
        print('xaa = {}'.format(_x[1].shape))
        x = self.classifier(x)
        return x
    
    
def evaluate(net, dataIterator, context):
    dataIterator.reset()
    loss = gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    for i, batch in enumerate(dataIterator):
        data =  batch.data[0].as_in_context(context)
        label = batch.data[1].as_in_context(context)
        output = net(data)
        L = loss(output, label)
        pred = (output > 0.5).reshape((-1,))
        #print('cor = {}'.format(pred == label))
        total_L += L.sum().asscalar()
        total_sample_num += len(label)
        total_correct_num += (pred == label).sum().asscalar()
        #print('total_correct_num = {}, total_correct_num = {}'.format(total_correct_num, total_sample_num))
        if (i + 1) % log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, dataIterator.num_data//dataIterator.batch_size,
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()
    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc


def train(model, train_data_iter, test_data_iter, trainer, loss, n_epoch, context):
    for epoch in tqdm_notebook(range(n_epoch), desc = 'epoch'):
        ## Training
        train_data_iter.reset()
        # Epoch training stats
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0

        for i, batch in enumerate(train_data_iter):
            _data = batch.data[0].as_in_context(context)
            _label = batch.data[1].as_in_context(context)
            L = 0
            wc = len(_data)
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += _data.shape[1]
            epoch_sent_num += _data.shape[1]
            with autograd.record():
                _out = model(_data)
                L = L + loss(_out, _label).mean().as_in_context(context)
            L.backward()
            trainer.step(_data.shape[0])
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            if (i + 1) % log_interval == 0:
                tqdm.write('[Epoch {} Batch {}/{}] elapsed {:.2f} s, \
                        avg loss {:.6f}, throughput {:.2f}K wps'.format(
                        epoch, i + 1, train_data.num_data//train_data.batch_size,
                        time.time() - start_log_interval_time,
                        log_interval_L / log_interval_sent_num,
                        log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        test_avg_L, test_acc = evaluate(model, test_data_iter, context)
        tqdm.write('[Epoch {}] train avg loss {:.6f}, valid acc {:.2f}, \
            valid avg loss {:.6f}, throughput {:.2f}K wps'.format(
            epoch, epoch_L / epoch_sent_num,
            test_acc, test_avg_L, epoch_wc / 1000 /
            (end_epoch_time - start_epoch_time)))
        
        
def get_attention(_att, _sentence, sentence_id):
    x = _sentence[sentence_id]
    _att = _att[sentence_id].asnumpy()
    word = []
    w_idx = []
    for token in x:
        #print(token)
        _word = idx2word[token]
        word.append(_word)
    att = pd.DataFrame(_att, index = word, columns = word)
    print(word)
    w_idx = [x for x in word if x is not 'PAD']
    print(w_idx)
    res = att.loc[w_idx][w_idx]
    return res


def prepare_data(file_name):
    nlp = spacy.load("en")
    word_freq = collections.Counter()
    max_len = 0
    num_rec = 0
    print('Count words and build vocab...')
    with open(file_name, 'rb') as f:
        for line in f:
            _lab, _sen = line.decode('utf8').strip().split('\t')
            words = [token.lemma_ for token in nlp(_sen) if token.is_alpha] # Stop word제거 안한 상태 
            # 제거를 위해 [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
            if len(words) > max_len:
                max_len = len(words)
            for word in words:
                word_freq[word] += 1
            num_rec += 1

    # most_common output -> list
    word2idx = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(MAX_VOCAB - 2))}
    word2idx ['PAD'] = 0
    word2idx['UNK'] = 1

    idx2word= {i:v for v, i in word2idx.items()}
    vocab_size = len(word2idx)

    print('Prepare data...')
    y = []
    x = []
    origin_txt = []
    with open(file_name, 'rb') as f:
        for line in f:
            _label, _sen = line.decode('utf8').strip().split('\t')
            origin_txt.append(_sen)
            y.append(int(_label))
            words = [token.lemma_ for token in nlp(_sen) if token.is_alpha] # Stop word제거 안한 상태
            words = [x for x in words if x != '-PRON-'] # '-PRON-' 제거
            _seq = []
            for word in words:
                if word in word2idx.keys():
                    _seq.append(word2idx[word])
                else:
                    _seq.append(word2idx['UNK'])
            if len(_seq) < MAX_SENTENCE_LENGTH:
                _seq.extend([0] * ((MAX_SENTENCE_LENGTH) - len(_seq)))
            else:
                _seq = _seq[:MAX_SENTENCE_LENGTH]
            x.append(_seq)

    pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index()
    return x, y ,origin_txt