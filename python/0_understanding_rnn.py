import numpy as np
np.set_printoptions(precision=3)

import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import nn, rnn

## Understanding Vanilla RNN

## Paramteres 
batch_size = 16
n_hidden_state = 10
embedding_input = 5


model = rnn.RNN(n_hidden_state, 1, layout = 'NTC', input_size = embedding_input \
              , prefix='mdl_')
model.collect_params().initialize(mx.init.Xavier(), ctx = mx.cpu())
initial_state = model.begin_state(batch_size = batch_size)

### Hiddens state size
print(model.params['mdl_l0_h2h_weight'].data().shape)
print(model.params['mdl_l0_h2h_bias'].data().shape)
print(model.params['mdl_l0_i2h_weight'].data().shape)
print(model.params['mdl_l0_i2h_bias'].data().shape)


### $T=1$
# Goes only 1 time-step
time_step = 1
dat = nd.random.normal(shape =(batch_size, time_step, embedding_input))
out, state = model(dat, initial_state)

out[0][0] == state[0][0][0]

h2h_weight = model.params['mdl_l0_h2h_weight'].data()
h2h_bias = model.params['mdl_l0_h2h_bias'].data()
i2h_weight = model.params['mdl_l0_i2h_weight'].data()
i2h_bias = model.params['mdl_l0_i2h_bias'].data()

res = nd.relu(nd.dot(dat[0][0], i2h_weight, transpose_b = True) + i2h_bias \
      + nd.dot(h2h_weight, initial_state[0][0][0]) + h2h_bias)

print('Compare out and state for t = 1')
print(res.asnumpy())
print(out[0][0].asnumpy())
print(state[0][0][0].asnumpy())

### $T=2$

time_step = 2
dat = nd.random.normal(shape =(batch_size, time_step, embedding_input))
out, state = model(dat, initial_state)

print('Compare out and state for t = 1 when T = 2')
print(out[0][1].asnumpy())
print(state[0][0][0].asnumpy())

h2h_weight = model.params['mdl_l0_h2h_weight'].data()
h2h_bias = model.params['mdl_l0_h2h_bias'].data()
i2h_weight = model.params['mdl_l0_i2h_weight'].data()
i2h_bias = model.params['mdl_l0_i2h_bias'].data()

out_t1 = nd.relu(nd.dot(dat[0][0], i2h_weight, transpose_b = True) + i2h_bias \
      + nd.dot(h2h_weight, initial_state[0][0][0]) + h2h_bias)

print('Compare calculation and out for t =1 when T = 2')
print(out_t1.asnumpy())
print(out[0][0].asnumpy())

out_t2 = nd.relu(nd.dot(dat[0][1], i2h_weight, transpose_b = True) + i2h_bias \
      + nd.dot(h2h_weight, out_t1) + h2h_bias)

print('Compare calculation, out, and state for t =2 when T = 2')
print(out_t2.asnumpy()) # calculation
print(state[0][0][0].asnumpy()) # State at last time step
print(out[0][1].asnumpy()) # Last time step


## Bidirectional LSTM

model = rnn.RNN(n_hidden_state, 1, layout = 'NTC', input_size = embedding_input, bidirectional = True, prefix='mdl_')
model.collect_params().initialize(mx.init.Xavier(), ctx = mx.cpu())
initial_state = model.begin_state(batch_size = 16) # list of length 1 with shape (2, 16, 10)

model.params

# Goes only 1 time-step
batch_size = 16
time_step = 2
dat = nd.random.normal(shape =(batch_size, time_step, embedding_input))
out, state = model(dat, initial_state)

# hidden state at t=1 (left, right) concatenated
out[0][0]


h2h_l_weight = model.params['mdl_l0_h2h_weight'].data()
h2h_l_bias = model.params['mdl_l0_h2h_bias'].data()
i2h_l_weight = model.params['mdl_l0_i2h_weight'].data()
i2h_l_bias = model.params['mdl_l0_i2h_bias'].data()
h2h_r_weight = model.params['mdl_r0_h2h_weight'].data()
h2h_r_bias = model.params['mdl_r0_h2h_bias'].data()
i2h_r_weight = model.params['mdl_r0_i2h_weight'].data()
i2h_r_bias = model.params['mdl_r0_i2h_bias'].data()

### $T = 1$

out_t1_l = nd.relu(nd.dot(dat[0][0], i2h_l_weight, transpose_b = True) + i2h_l_bias \
      + nd.dot(h2h_l_weight, initial_state[0][0][0]) + h2h_l_bias)
print('RNN from left at t = 1 when T = 2')
print(out_t1_l)


out_t2_r = nd.relu(nd.dot(dat[0][1], i2h_r_weight, transpose_b = True) + i2h_r_bias \
      + nd.dot(h2h_r_weight, initial_state[0][0][1]) + h2h_r_bias)
out_t1_r = nd.relu(nd.dot(dat[0][0], i2h_r_weight, transpose_b = True) + i2h_r_bias \
      + nd.dot(h2h_r_weight, out_t2_r) + h2h_r_bias)
print('RNN from right at t = 1 when T = 2')
print(out_t2_r)
print(out_t1_r)

print('Compare calculation and output for bidirectional RNN at t = 1 when T = 2')
print(out[0][0].asnumpy())
print(nd.concat(out_t1_l, out_t1_r, dim = 0).asnumpy())

### $T = 2$

out_t2_l = nd.relu(nd.dot(dat[0][1], i2h_l_weight, transpose_b = True) + i2h_l_bias \
      + nd.dot(h2h_l_weight, out_t1_l) + h2h_l_bias)
print('RNN from left at t = 2 when T = 2')
print(out_t2_l)

print('Compare calculation and output for bidirectional RNN at t = 2 when T = 2')
print(out[0][1].asnumpy())
print(nd.concat(out_t2_l, out_t2_r, dim = 0).asnumpy())

print('Compare calculation and state for bidirectional RNN when T = 2')
print(nd.concat(out_t2_l, out_t1_r, dim = 0).asnumpy())
print(nd.concat(state[0][0][0], state[0][1][0], dim = 0).asnumpy())


## Stacking

model = rnn.RNN(n_hidden_state, 2, layout = 'NTC' \
              , input_size = embedding_input, bidirectional = False, prefix='mdl_')
model.collect_params().initialize(mx.init.Xavier(), ctx = mx.cpu())
initial_state = model.begin_state(batch_size = batch_size) # list of length 1 with shape (2, 16, 10)

i2h_weight_0 = model.params['mdl_l0_i2h_weight'].data()
h2h_weight_0 = model.params['mdl_l0_h2h_weight'].data()
i2h_bias_0 = model.params['mdl_l0_i2h_bias'].data()
h2h_bias_0 = model.params['mdl_l0_h2h_bias'].data()
i2h_weight_1 = model.params['mdl_l1_i2h_weight'].data()
h2h_weight_1 = model.params['mdl_l1_h2h_weight'].data()
i2h_bias_1 = model.params['mdl_l1_i2h_bias'].data()
h2h_bias_1 = model.params['mdl_l1_h2h_bias'].data()

# Goes only 1 time-step
time_step = 2
dat = nd.random.normal(shape =(batch_size, time_step, embedding_input))
out, state = model(dat, initial_state)

print('Output of layer 1 at t = 1 and t = 2')
print(out[0][0].asnumpy()) # Layer 1 at time 1
print(out[0][1].asnumpy()) # Layer 1 at time 2

print('State of layer 0 and 1 at t = 2')
print(state[0][0][0].asnumpy()) # Layer 0 at time 2
print(state[0][1][0].asnumpy()) # Layer 1 at time 2

### $T=1$

out_0_t1 = nd.relu(nd.dot(dat[0][0], i2h_weight_0, transpose_b = True) + i2h_bias_0 \
      + nd.dot(h2h_weight_0, initial_state[0][0][0]) + h2h_bias_0)
out_1_t1 = nd.relu(nd.dot(out_0_t1, i2h_weight_1, transpose_b = True) + i2h_bias_1 \
      + nd.dot(h2h_weight_1, initial_state[0][1][0]) + h2h_bias_1)

print('Calculation for layer 1 and 2 at t = 1')
print(out_0_t1.asnumpy())
print(out_1_t1.asnumpy())

### $T=2$

out_0_t2 = nd.relu(nd.dot(dat[0][1], i2h_weight_0, transpose_b = True) + i2h_bias_0 \
      + nd.dot(h2h_weight_0, out_0_t1) + h2h_bias_0)
out_1_t2 = nd.relu(nd.dot(out_0_t2, i2h_weight_1, transpose_b = True) + i2h_bias_1 \
      + nd.dot(h2h_weight_1, out_1_t1) + h2h_bias_1)

print('Calculation for layer 1 and 2 at t = 2')
print(out_0_t2.asnumpy())
print(out_1_t2.asnumpy())
