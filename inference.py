from keras import layers
import tensorflow as tf
from Resnet import ResNet50
import keras
from keras.models import Input,Model

def inference(image_pre,image_now,location_tensor,shape):

    #extra high leavl feature
    input = Input(shape=shape)
    vision_model = keras.applications.MobileNet(include_top=False,
                                                weights='imagenet',
                                                input_tensor=input,
                                                input_shape=[224,224,3])
    vision_model.trainable = False
    feature1 = vision_model(image_pre)
    feature2 = vision_model(image_now)

    #reshape tensor to vector
    flatten = layers.Flatten()
    feature1 = flatten(feature1)
    feature2 = flatten(feature2)

    #get high level feature
    fc_unit = 512
    fc1 = layers.Dense(units=fc_unit,name='fc1',activation='relu')
    reshape = layers.Reshape((1,fc_unit))
    bn1 = layers.BatchNormalization(name='bn1')

    #fc1 block
    feature1 = fc1(feature1)
    feature2 = fc1(feature2)
    feature1 = bn1(feature1)
    feature2 = bn1(feature2)
    feature1 = layers.Activation('relu')(feature1)
    feature2 = layers.Activation('relu')(feature2)

    feature1 = reshape(feature1)
    feature2 = reshape(feature2)

    #build feature to (samle,time_step,input_dim)
    out = layers.concatenate([feature1,feature2],axis=1)

    out = layers.GRU(units=128,name='GRU')(out)
    out = layers.concatenate([out,location_tensor],axis=1)

    out = layers.Dense(units=64, name='fc2')(out)
    out = layers.Activation('relu')(out)
    out = layers.Dense(units=4,name='fc3')(out)
    out = layers.Activation('sigmoid')(out)

    return out
