from keras import layers
import tensorflow as tf
from Resnet import ResNet50
import keras
from keras.models import Input,Model
import keras.backend as K

def inference(image_pre,image_now,location_tensor):

    #extra high leavl feature
    input = Input(shape=[224,224,3])
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
    fc1_unit = 256
    fc1 = layers.Dense(units=fc1_unit,name='fc1',activation='relu')
    bn1 = layers.BatchNormalization(name='bn1')

    #fc1 block
    feature1 = fc1(feature1)
    feature2 = fc1(feature2)
    feature1 = bn1(feature1)
    feature2 = bn1(feature2)
    feature1 = layers.Activation('relu')(feature1)
    feature2 = layers.Activation('relu')(feature2)

    #union_info layer
    union_feature = layers.concatenate([feature1,location_tensor],axis=1)
    feature1 = layers.Dense(units=fc1_unit,name='union_block')(union_feature)
    feature1 = layers.BatchNormalization()(feature1)
    feature1 = layers.Activation('relu')(feature1)

    #define reshape layer
    reshape_unit = fc1_unit
    reshape = layers.Reshape((1,reshape_unit))

    feature1 = reshape(feature1)
    feature2 = reshape(feature2)

    #build feature to (samle,time_step,input_dim)
    out = layers.concatenate([feature1,feature2],axis=1)

    #get fix_location
    out = layers.GRU(units=8,name='fix_GRU',activation='tanh')(out)

    out = layers.add([out,location_tensor])

    return out
