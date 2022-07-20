# Main snippet on computing the lower bound derived in Eq. 3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import mse, BinaryCrossentropy

############### Interpretible ##################

"""
Neural time series decomposition function:

Inputs: Subsequences[Batch_size, length_of_subsequence, num_of_feature]

W: frequency;
p: phase;

"""
class T2V(Layer):

    def __init__(self, output_dim=None, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                shape=(1,10,1),
                                initializer='uniform',
                                trainable=True)

        self.P = self.add_weight(name='P',
                                shape=(1,10,1),
                                initializer='uniform',
                                trainable=True)

        self.w = self.add_weight(name='w',
                                shape=(10,1),
                                initializer='uniform',
                                trainable=True)

        self.p = self.add_weight(name='p',
                                shape=(10,1),
                                initializer='uniform',
                                trainable=True)

        super(T2V, self).build(input_shape)

    def call(self, x):
        original = self.w*x + self.p
        x = K.repeat_elements(x, self.output_dim, -1)
        #x_ = K.mean(x, axis = -1,keepdims = True)
        sin_trans = K.sin(x * self.W + self.P)
        return K.concatenate([sin_trans,original], -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim+1)

############### Task-agnoistic ##################
"""
Function f in article: network_encoder + T2V + network_autoregressive

Function g_1, g_2 in article: network_prediction

"""

def network_encoder(x, latent_dim=latent_dim):
    x = Dense(units= 1, activation = 'linear', name = 'first_layer')(x)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.LeakyReLU()(x)

    #x = keras.layers.Dense(units= 15, activation = 'linear', name = 'second_layer')(x)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.LeakyReLU()(x)

    #x = keras.layers.Dense(units=latent_dim, activation='linear', name='encoder_embedding')(x)

    return x
## define the network integrates the information along the sequence
## zt- gar -> ct

def network_autoregressive(x):  ## to get Ct variable via RNN cell; GRU

  x = GRU(units = 1,
          return_sequences=False, name = 'ar_context')(x)

  return x

## define mapping Ct -> other z_t+1, z_t+2

def network_prediction(context, latent_dim, predict_terms):

  outputs = []

  for i in range (predict_terms):
    outputs.append(Dense(units=latent_dim, activation="linear", name='z_t_{i}'.format(i=i))(context))
  if len(outputs) == 1:
        output =Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
  else:
        output =Lambda(lambda x: K.stack(x, axis=1))(outputs)

  return output

############### Sample code of running unified local predicitve model ##################

### Defining the alpha
alpha = 0.0001

### Set-ups for data
X_shape = (1, )


##### Function f #####

encoder_input = Input(X_shape)
encoder_output = network_encoder(encoder_input, latent_dim)
encoder_model = Model(encoder_input, encoder_output, name = 'encoder')
terms = 10  ## length_of_subsequence
predict_terms = 10

x_input = Input((terms, n_features)) ## x_t-3,...,x_t
x_encoded_sequence = TimeDistributed(encoder_model)(x_input) ## z_t-3,...,z_t

context = T2V(1, 10)(x_encoded_sequence)  ## Neural decomposition function
context = network_autoregressive(context) ## produce Ct variable

##### Function g_1 #####
dec = Dense(1)(context)
preds = network_prediction(context, latent_dim, predict_terms)  ## predict_terms: num of steps for prediction z_t+1,..., z_t+4 prediction based on previous values
preds_context = T2V(1,10)(preds)


##### g_2 #####
y_input =  Input((predict_terms, n_features )) ## x_t+1,..., x_t+4
y_input_ =  Input(X_shape)
#y_input_per = T2V(1,1)(y_input_)
#y_context = T2V(1,10)(y_encoded)
y_encoded = TimeDistributed(encoder_model)(y_input) ## z_t+1,..., z_t+4; true values based on incoming input


##### Loss computed for g_1 #####
MSE = (mse(dec,y_input_))

##### Loss computed for g_2 #####

dot_product = K.mean(y_encoded * preds, axis=-1)##
dot_product = K.mean(dot_product, axis = -1, keepdims = True)  ## avearge overall all prediction steps
dot_product_probs = K.sigmoid(dot_product)

##### Loss g_1 + g_2 #####
loss = alpha *(MSE) + (1- alpha) *(dot_product_probs)


##### Model training #####
ULP = Model([x_input, y_input, y_input_],[preds, dec])

ULP.add_loss(loss)
    #cpc_extra.add_loss(MSE)
ULP.compile(optimizer ='adam')

ULP.fit([X[:-1],X[1:], y[:-1]],
         verbose=0, batch_size=256, epochs=200,
         shuffle=True)
