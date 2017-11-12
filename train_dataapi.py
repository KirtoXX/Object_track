from keras.models import Model
from inference2 import inference
import tensorflow as tf
from keras import layers
from keras import backend as K
from keras import metrics
import tensorflow as tf
import numpy as np
from tfrecord import fiel_to_tensor

def main():
    batch_size = 32
    buffer_size = 1000
    epoch = 100
    input_shape = [224,224,3]

    x1, x2, y1, y2 = fiel_to_tensor()

    image_pre = layers.Input(name='image1', tensor=x1)
    image_now = layers.Input(name='image2', tensor=x2)
    location = layers.Input(name='location', tensor=y1)

    logit = inference(x1,x2,y1)
    model = Model(inputs=[image_pre,image_now,location],outputs=logit)

    model.compile(optimizer='RMSprop',
                  loss='mse',
                  metrics=[metrics.mean_squared_error],
                  target_tensors=[y2])
    model.summary()
    sess = K.get_session()
    summary = tf.summary.FileWriter('log/',sess.graph)

    #total image=21395
    i = np.ceil(21395/batch_size).astype('int')

    model.fit(epochs=300,steps_per_epoch=i)

if __name__ == '__main__':
    main()