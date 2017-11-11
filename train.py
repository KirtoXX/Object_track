from keras.models import Model
from inference import inference
import tensorflow as tf
from keras.models import Input
from keras import backend as K
from keras import metrics
import tensorflow as tf
import numpy as np

def main():
    input_shape = [224,224,3]
    image_pre = Input(name='image1',shape=input_shape)
    image_now = Input(name='image2',shape=input_shape)
    location = Input(name='location',shape=[4])

    logit = inference(image_pre,image_now,location,shape=input_shape)

    model = Model(inputs=[image_pre,image_now,location],outputs=logit)
    model.compile(optimizer='RMSprop',
                  loss='mse',
                  metrics=[metrics.mean_squared_error])
    model.summary()
    sess = K.get_session()
    summary = tf.summary.FileWriter('log/',sess.graph)

    data1 = np.load('npy_data/data1.npy')
    data2 = np.load('npy_data/data2.npy')
    y1 = np.load('npy_data/y1.npy')
    y2 = np.load('npy_data/y2.npy')

    model.fit(x=[data1,data2,y1],y=y2,batch_size=32,epochs=300)

if __name__ == '__main__':
    main()