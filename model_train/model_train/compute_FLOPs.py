import tensorflow.keras as keras
import tensorflow as tf
import keras.backend as K
from utils import count_model_params_flops

#inp_shape = [96, 96, 1]
#inp_shape = [128, 128, 3]
inp_shape = [28, 28, 41]

def identity_block_v1(x, kernel_num, kernel_size, stride=[1,1], padding='same', mode='selu'):
    inp = x # channel=32
    x = keras.layers.Conv2D(kernel_num, kernel_size, stride, padding=padding, activation=mode)(x)
    x = keras.layers.Conv2D(kernel_num, kernel_size, stride, padding=padding, activation=mode)(x)
    out = keras.layers.add([x, inp])
    return out

def conv_block(x, kernel_num, kernel_size, stride=[2,2], padding='same', mode='selu'):
    inp = x # channel=32
    x = keras.layers.Conv2D(kernel_num, kernel_size, stride, padding=padding, activation=mode)(x)
    x = keras.layers.Conv2D(kernel_num, kernel_size, [1,1], padding=padding, activation=mode)(x)
    d = keras.layers.Conv2D(kernel_num, kernel_size, stride, padding=padding, activation=mode)(inp)

    out = keras.layers.add([x, d])
    return out

def resnet():
    channel = 32
    stage = [1,1,1,1,1]

    inp = keras.layers.Input(inp_shape)
    x = keras.layers.Conv2D(channel, [3,3], [1,1], 
                            padding='same', activation='selu')(inp)
    for s in range(len(stage)):
        for i in range(stage[s]):
            x = identity_block_v1(x, channel, [3,3])
        if s < len(stage)-1:
            x = conv_block(x, channel, [3,3])

    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(2)(x)

    return keras.models.Model(inputs=inp, outputs=out)

def cnn_6():
    inp = keras.layers.Input(inp_shape)
    x = keras.layers.Conv2D(32, [3,3], [1,1], 
                            padding='same', activation='selu')(inp)
    x = keras.layers.Conv2D(32, [3,3], [2,2], 
                            padding='same', activation='selu')(x)
    x = keras.layers.Conv2D(64, [3,3], [1,1], 
                            padding='same', activation='selu')(x)
    x = keras.layers.Conv2D(64, [3,3], [2,2], 
                            padding='same', activation='selu')(x)
    x = keras.layers.Conv2D(128, [1,1], [1,1], 
                            padding='same', activation='selu')(x)
    #x = keras.layers.Conv2D(256, [5,5], [2,2], 
    #                        padding='same', activation='selu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    out = keras.layers.Dense(2)(x)

    return keras.models.Model(inputs=inp, outputs=out)



def main():

    m = cnn_6()
    m.summary()
    param, FLOPs = count_model_params_flops(m)
if __name__ == '__main__':
    main()