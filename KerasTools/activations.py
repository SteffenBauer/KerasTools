import keras.backend as K

def gelu(x):
    return 0.5*x*(1.0+K.tanh(0.797884561 * (x+0.044715*K.pow(x,3))))

