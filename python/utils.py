import collections, spacy, time
import pandas as pd
import mxnet as mx
import numpy as np

def prepare_data(file_name, max_sen_len, max_vocab):
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
    word2idx = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(max_vocab - 2))}
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
            if len(_seq) < max_sen_len:
                _seq.extend([0] * ((max_sen_len) - len(_seq)))
            else:
                _seq = _seq[:max_sen_len]
            x.append(_seq)

    pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index()
    return x, y ,origin_txt, idx2word

def one_hot(x, vocab_size):
    res = np.zeros(shape = (vocab_size))
    res[x] = 1
    return res

def evaluate(net, log_interval, dataIterator, context):
    from mxnet import gluon
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
        total_L += L.sum().asscalar()
        total_sample_num += len(label)
        total_correct_num += (pred == label).sum().asscalar()
        if (i + 1) % log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, dataIterator.num_data//dataIterator.batch_size,
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()
    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc


def train(n_epoch, log_interval, model, train_data, valid_data, trainer, loss, context = mx.cpu()):
    import time, re
    from tqdm import tqdm
    from mxnet import autograd
    
    for epoch in tqdm(range(n_epoch), desc = 'epoch'):
        ## Training
        train_data.reset()
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

        for i, batch in enumerate(train_data):
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
        test_avg_L, test_acc = evaluate(model, log_interval, valid_data, context)
        tqdm.write('[Epoch {}] train avg loss {:.6f}, valid acc {:.2f}, \
            valid avg loss {:.6f}, throughput {:.2f}K wps'.format(
            epoch, epoch_L / epoch_sent_num,
            test_acc, test_avg_L, epoch_wc / 1000 /
            (end_epoch_time - start_epoch_time)))
        
### Prediction
def get_pred(net, loss, iterator, idx2word, context):
    from mxnet import nd
    pred_sa = []
    label_sa = []
    va_text = []
    iterator.reset()
    for i, batch in enumerate(iterator):
        if i % 100 == 0:
            print('i = {}'.format(i))
        data =  batch.data[0].as_in_context(context)
        label = batch.data[1].as_in_context(context)
        output = net(data)
        L = loss(output, label)
        pred = (nd.sigmoid(output) > 0.5).reshape((-1,))
        pred_sa.extend(pred.asnumpy())
        label_sa.extend(label.asnumpy())
        va_text.extend([' '.join([idx2word[np.int(x)] for x in y.asnumpy() if idx2word[np.int(x)] is not 'PAD']) for y in data])
    pred_sa_pd = pd.DataFrame(pred_sa, columns  = ['pred_sa'])
    label_pd = pd.DataFrame(label_sa, columns = ['label'])
    text_pd = pd.DataFrame(va_text, columns = ['text'])
    res = pd.concat([text_pd, pred_sa_pd, label_pd], axis = 1)
    return res


### Testset
def gen_n_test(N):
    q = []
    y = []
    for i in range(N):
        a, b = n(), n() 
        question = '{}+{}'.format(a, b)
        answer_y = str(a + b)
        q.append(question)
        y.append(answer_y)
    return(q,y)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
    
def n(digits =3):
    number = ''
    for i in range(np.random.randint(1, digits + 1)):
        number += np.random.choice(list('0123456789'))
    return int(number)


def padding(chars, maxlen):
    return chars + ' ' * (maxlen - len(chars))


def train_seq2seq(epochs, log_interval, model, train_data, valid_data, trainer, loss, char_indices, indices_char, ctx = mx.cpu()):
    from mxnet import autograd, nd
    tot_train_loss = []
    tot_va_loss = []
    for e in range(epochs):
        train_loss = []
        for i, (x_data, y_data, z_data) in enumerate(train_data):
            x_data = x_data.as_in_context(ctx).astype('float32')
            y_data = y_data.as_in_context(ctx).astype('float32')
            z_data = z_data.as_in_context(ctx).astype('float32')

            with autograd.record():
                z_output = model(x_data, y_data)
                loss_ = loss(z_output, z_data)
            loss_.backward()
            trainer.step(x_data.shape[0])
            curr_loss = nd.mean(loss_).asscalar()
            train_loss.append(curr_loss)

        if e % log_interval == 0:
            q, y = gen_n_test(10)
            for i in range(10):
                with autograd.predict_mode():
                    p = model.calculation(q[i], char_indices, indices_char).strip()
                    iscorr = 1 if p == y[i] else 0
                    if iscorr == 1:
                        print(colors.ok + '☑' + colors.close, end=' ')
                    else:
                        print(colors.fail + '☒' + colors.close, end=' ')
                    print("{} = {}({}) 1/0 {}".format(q[i], p, y[i], str(iscorr) ))
        #caculate test loss
        va_loss = calculate_loss(model, valid_data, loss_obj = loss, ctx=ctx) 

        print("Epoch %s. Train Loss: %s, Test Loss : %s" % (e, np.mean(train_loss), va_loss))    
        tot_va_loss.append(va_loss)
        tot_train_loss.append(np.mean(train_loss))
    return tot_train_loss, tot_va_loss


def calculate_loss(model, data_iter, loss_obj, ctx = mx.cpu()):
    from mxnet import nd, autograd
    test_loss = []
    for i, (x_data, y_data, z_data) in enumerate(data_iter):
        x_data = x_data.as_in_context(ctx).astype('float32')
        y_data = y_data.as_in_context(ctx).astype('float32')
        z_data = z_data.as_in_context(ctx).astype('float32')
        with autograd.predict_mode():
            z_output = model(x_data, y_data)
            loss_te = loss_obj(z_output, z_data)
        curr_loss = nd.mean(loss_te).asscalar()
        test_loss.append(curr_loss)
    return np.mean(test_loss)