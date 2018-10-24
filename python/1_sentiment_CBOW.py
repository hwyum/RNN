import nltk
import os
import pandas as pd
import numpy as np
import nltk
import collections
from sklearn.preprocessing import normalize
from utils import *

nltk.download('punkt')
#os.system('python -m spacy download en')

max_vocab = 2000
max_sen_len = 40

x, y, origin_txt = prepare_data('../data/umich-sentiment-train.txt', max_sen_len, max_vocab)

pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index()

## Sentence representation: Average of BOW
x_1 = np.array([np.sum(np.array([one_hot(word, max_vocab) for word in example]), axis = 0) for example in x])

## Data process - tr/va split and define iterator
tr_idx = np.random.choice(range(x_1.shape[0]), int(x_1.shape[0] * .8))
va_idx = [x for x in range(x_1.shape[0]) if x not in tr_idx]

tr_x = x_1[tr_idx, :]
tr_y = [y[i] for i in tr_idx]
va_x = x_1[va_idx, :]
va_y = [y[i] for i in va_idx]


## Classification
### XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

xgb = XGBClassifier()
xgb.fit(tr_x, tr_y)

y_pred_xgb = xgb.predict(va_x)
pred_xgb = [round(val) for val in y_pred_xgb]

# Check predictions
#pred_pd= pd.DataFrame(pred_xgb, columns = ['pred']).reset_index()
#pred_pd.groupby(['pred']).count()

accuracy_xgb = accuracy_score(va_y, pred_xgb)
print('Accuracy: %.2f%%'%(accuracy_xgb * 100.0))

### Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(tr_x, tr_y)

y_pred_rf = rf.predict(va_x)
pred_rf = [round(val) for val in y_pred_rf]

accuracy_rf = accuracy_score(va_y, pred_rf)
print('Accuracy: %.2f%%'%(accuracy_rf * 100.0))

### SVM

from sklearn import svm

models = (svm.SVC(kernel = 'linear', C = 1.0), # C: SVM Regularization parameter
          svm.LinearSVC(C = 1.0),
          svm.SVC(kernel = 'rbf', gamma = .7, C = 1.0),
          svm.SVC(kernel = 'poly', degree = 3, C = 1.0)
)

models = (mdl.fit(tr_x, tr_y) for mdl in models)

y_pred_svm = (mdl.predict(va_x) for mdl in models)
pred_svm = [[round(val) for val in _pred] for _pred in y_pred_svm]

accuracy_svm = [accuracy_score(va_y, pred) for pred in pred_svm]
print('Accuracy: {}'.format(np.round(accuracy_svm, 4)*100))

## Check results

va_txt = pd.DataFrame(np.array([origin_txt[idx] for idx in va_idx]), columns = ['txt'])
pred_rf_pd = pd.DataFrame(pred_rf, columns  = ['pred_rf'])
pred_xgb_pd = pd.DataFrame(pred_xgb, columns  = ['pred_xgb'])
pred_svm_svc_pd = pd.DataFrame(pred_svm[2], columns  = ['pred_svm'])
label_pd = pd.DataFrame(va_y, columns = ['label'])
result = pd.concat([va_txt, pred_rf_pd, pred_xgb_pd, pred_svm_svc_pd, label_pd], axis = 1)

result.head()

print('# of error case from RF : {}'.format(result[result['pred_rf'] != result['label']].shape[0]))
print('# of error case from XGB: {}'.format(result[result['pred_xgb'] != result['label']].shape[0]))
print('# of error case from SVM: {}'.format(result[result['pred_svm'] != result['label']].shape[0]))


### DNN with embedding layer

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
context = mx.cpu()

class MLP(nn.Block):
    def __init__(self, input_dim, emb_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.embed = nn.Embedding(input_dim = input_dim, output_dim = emb_dim)
            self.dense1 = nn.Dense(64)
            #self.dense2 = nn.Dense(32, activation = 'relu')
            self.bn = nn.BatchNorm()
            self.dense2 = nn.Dense(2)
            
    def forward(self, x):
        x = self.embed(x)
        x = self.dense1(x)
        x = self.bn(x)
        x = nd.relu(x)
        x = self.dense2(x)
        return x

def acc_f(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    #print('pred = {}'.format(pred))
    #print('label = {}'.format(label))
    corr = ((pred > 0.5) == label)*1.
    return (((pred > 0.5) == label)*1.).mean()
tr_metric = mx.metric.CustomMetric(acc_f)
va_metric = mx.metric.CustomMetric(acc_f)

n_epoch = 10
batch_size = 64
from tqdm import tqdm, tqdm_notebook
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False)
valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False)

mlp = MLP(input_dim = max_vocab, emb_dim = 50)
mlp.collect_params().initialize(mx.init.Xavier(), ctx = context)
loss = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(mlp.collect_params(), 'adam', {'learning_rate': 1e-3})

for epoch in tqdm_notebook(range(n_epoch), desc = 'epoch'):
    ## Training
    train_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(train_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        with autograd.record():
            _out = mlp(_dat)
            _los = nd.sum(loss(_out, _label)) # 배치의 크기만큼의 loss가 나옴
            _los.backward()
        trainer.step(_dat.shape[0])
        n_obs += _dat.shape[0]
        #print(n_obs)
        _total_los += nd.sum(_los).asnumpy()
        # Epoch loss를 구하기 위해서 결과물을 계속 쌓음
        pred.extend(nd.softmax(_out)[:,1].asnumpy()) # 두번째 컬럼의 확률이 예측 확률
        label.extend(_label.asnumpy())
    #print(pred)
    #print([round(p) for p in pred]) # 기본이 float임
    #print(label)
    #print('**** ' + str(n_obs))
    #print(label[:10])
    #print(pred[:10])
    #print([round(p) for p in pred][:10])
    tr_acc = accuracy_score(label, [round(p) for p in pred])
    tr_loss = _total_los/n_obs
    
    ### Evaluate training
    valid_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(valid_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        _out = mlp(_dat)
        _pred_score = nd.softmax(_out)
        n_obs += _dat.shape[0]
        _total_los += nd.sum(loss(_out, _label)).asnumpy()
        pred.extend(nd.softmax(_out)[:,1].asnumpy())
        label.extend(_label.asnumpy())
    va_acc = accuracy_score(label, [round(p) for p in pred])
    va_loss = _total_los/n_obs
    tqdm.write('Epoch {}: tr_loss = {}, tr_acc= {}, va_loss = {}, va_acc= {}'.format(epoch, tr_loss, tr_acc, va_loss, va_acc))

y_pred_mlp = mlp(nd.array(va_x, ctx = context))
# softmax를 적용하고
# 두번째 열을 뽑아와서
# nd.round 함수를 적용해서 0/1 예측값을 얻고
# numpy array로 바꾸고
# 첫번째 원소를 뽑아서 예측 label로 사용
pred_mlp = [nd.round(val).asnumpy()[0] for val in nd.softmax(y_pred_mlp)[:, 1]] 

accuracy_mlp = accuracy_score(va_y, pred_mlp)
print('Accuracy: %.2f%%'%(accuracy_mlp * 100.0))

#### DNN without embedding

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense1 = nn.Dense(64)
            #self.dense2 = nn.Dense(32, activation = 'relu')
            self.bn = nn.BatchNorm()
            self.dense2 = nn.Dense(2)
            
    def forward(self, x):
        x = self.dense1(x)
        x = self.bn(x)
        x = nd.relu(x)
        x = self.dense2(x)
        return x

n_epoch = 10
batch_size = 64
from tqdm import tqdm, tqdm_notebook

mlp_no_embedding = MLP()
mlp_no_embedding.collect_params().initialize(mx.init.Xavier(), ctx = context)
loss = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(mlp_no_embedding.collect_params(), 'adam', {'learning_rate': 1e-3})

for epoch in tqdm_notebook(range(n_epoch), desc = 'epoch'):
    ## Training
    train_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(train_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        with autograd.record():
            _out = mlp_no_embedding(_dat)
            _los = nd.sum(loss(_out, _label)) # 배치의 크기만큼의 loss가 나옴
            _los.backward()
        trainer.step(_dat.shape[0])
        n_obs += _dat.shape[0]
        #print(n_obs)
        _total_los += nd.sum(_los).asnumpy()
        # Epoch loss를 구하기 위해서 결과물을 계속 쌓음
        pred.extend(nd.softmax(_out)[:,1].asnumpy()) # 두번째 컬럼의 확률이 예측 확률
        label.extend(_label.asnumpy())
    #print(pred)
    #print([round(p) for p in pred]) # 기본이 float임
    #print(label)
    #print('**** ' + str(n_obs))
    #print(label[:10])
    #print(pred[:10])
    #print([round(p) for p in pred][:10])
    tr_acc = accuracy_score(label, [round(p) for p in pred])
    tr_loss = _total_los/n_obs
    
    ### Evaluate training
    valid_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(valid_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        _out = mlp(_dat)
        _pred_score = nd.softmax(_out)
        n_obs += _dat.shape[0]
        _total_los += nd.sum(loss(_out, _label)).asnumpy()
        pred.extend(nd.softmax(_out)[:,1].asnumpy())
        label.extend(_label.asnumpy())
    va_acc = accuracy_score(label, [round(p) for p in pred])
    va_loss = _total_los/n_obs
    tqdm.write('Epoch {}: tr_loss = {}, tr_acc= {}, va_loss = {}, va_acc= {}'.format(epoch, tr_loss, tr_acc, va_loss, va_acc))

y_pred_mlp_no_embedding = mlp_no_embedding(nd.array(va_x, ctx = context))
# softmax를 적용하고
# 두번째 열을 뽑아와서
# nd.round 함수를 적용해서 0/1 예측값을 얻고
# numpy array로 바꾸고
# 첫번째 원소를 뽑아서 예측 label로 사용
pred_mlp_no_embedding = [nd.round(val).asnumpy()[0] for val in nd.softmax(y_pred_mlp)[:, 1]] 

accuracy_mlp_no_embedding = accuracy_score(va_y, pred_mlp_no_embedding)
print('Accuracy: %.2f%%'%(accuracy_mlp_no_embedding * 100.0))

## Errors

va_txt = pd.DataFrame(np.array([origin_txt[idx] for idx in va_idx]), columns = ['txt'])
pred_mlp_no_embedding_pd = pd.DataFrame(pred_mlp_no_embedding, columns  = ['pred_mlp_no_embedding'])
label_pd = pd.DataFrame(va_y, columns = ['label'])
result = pd.concat([va_txt, pred_mlp_no_embedding_pd, label_pd], axis = 1)

result[result['pred_mlp_no_embedding'] != result['label']].shape

_pred_score[:, 0]
