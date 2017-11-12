import tensorflow as tf
import os
import numpy as np
import tensorflow as tf

def path_to_image(dir):
    file_names = os.listdir(dir)
    #build input iamge_pre
    temp = []
    for image in file_names:
        if image.find('.jpg') != -1:
            temp.append(dir+"/"+image)
    #updata_2list
    pre_names = temp.copy()
    del pre_names[-1]
    #build input image_now
    now_names = temp.copy()
    del now_names[0]
    print(len(pre_names)," ",len(now_names))
    #print(pre_names)
    return pre_names,now_names

def get_path():
    dir = 'vot2016/'
    file_dir = os.listdir(dir)
    pre_data = []
    now_data = []
    for i,name in enumerate(file_dir):
        path = dir+name
        print(path)
        data1,data2 = path_to_image(path)
        pre_data.extend(data1)
        now_data.extend(data2)
    print(len(pre_data))
    print(len(now_data))
    print(pre_data[0])
    print(now_data[0])
    return pre_data,now_data

def _parse_function(pre,now,y1,y2):
  pre_sting = tf.read_file(pre)
  pre_decoded = tf.image.decode_jpeg(pre_sting)
  pre_resized = tf.image.resize_images(pre_decoded, [224,224])
  now_string = tf.read_file(now)
  now_decoded = tf.image.decode_jpeg(now_string)
  now_resized = tf.image.resize_images(now_decoded, [224, 224])
  return pre_resized,now_resized,y1,y2

def fiel_to_tensor(buffer_size=1000,epoch=100,batch_size=32):
    pre, now = get_path()
    Y1 = np.load('npy_data/pre_lable.npy')
    Y2 = np.load('npy_data/now_lable.npy')
    dataset = tf.data.Dataset.from_tensor_slices((pre,now,Y1,Y2))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    #epoch
    dataset = dataset.repeat(epoch)
    #batchsize
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    x1,x2,y1,y2 = iterator.get_next()
    return x1,x2,y1,y2



def main():
    x1,x2,y1,y2 = fiel_to_tensor()

if __name__ == '__main__':
    main()


