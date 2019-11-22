import tensorflow.keras as keras
import tensorflow as tf
import keras.backend as K
from utils import count_model_params_flops

inp_shape = [96, 96, 1]
#inp_shape = [128, 128, 1]

def identity_block_v1(x, kernel_num, kernel_size, stride=[1,1], padding='same', mode='selu'):
    inp = x # channel=32
    x = keras.layers.Conv2D(kernel_num, kernel_size, stride, padding=padding, activation=mode)(x)
    x = keras.layers.Conv2D(kernel_num, kernel_size, stride, padding=padding, activation=mode)(x)
    out = keras.layers.add([x, inp])
    return out
def resnet():
    channel = 32
    stage = [1,1,1,1,1]

    inp = keras.layers.Input(inp_shape)
    x = keras.layers.Conv2D(channel, [7,7], [1,1], 
                            padding='same', activation='selu')(inp)
    for s in range(len(stage)):
        for i in range(stage[s]):
            x = identity_block_v1(x, channel, [3,3])
        if s < len(stage)-1:
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(2)(x)

    return keras.models.Model(inputs=inp, outputs=out)

def main():

    m = resnet()
    m.summary()
    param, FLOPs = count_model_params_flops(m)
if __name__ == '__main__':
    main()