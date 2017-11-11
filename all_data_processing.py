from __future__ import print_function
from scipy import misc
import numpy as np
import os

def to_npy(paths,dir):
    m = len(paths)
    npdata = np.zeros([m,224,224,3])
    for i,name in enumerate(paths):
        name = dir+paths[i]
        temp = misc.imread(name,mode='RGB')
        temp = misc.imresize(temp,[224,224])
        temp = np.expand_dims(temp,axis=0)
        npdata[i] = temp
    return npdata

def path_to_image(dir):
    dir = dir+'/img/'
    file_names = os.listdir(dir)
    #build input iamge_pre
    pre_names = file_names.copy()
    del pre_names[-1]
    #build input image_now
    now_names = file_names.copy()
    del now_names[0]

    print(len(pre_names))
    print(len(now_names))

    pre_data1 = to_npy(pre_names,dir)
    now_data2 = to_npy(now_names,dir)

    print(pre_data1.shape,now_data2.shape)
    return pre_data1,now_data2

def file2list(filename):
    fr = open(filename)
    array = fr.readlines() #以文件中的每行为一个元素，形成一个list列表
    num = len(array)
    returnMat = np.zeros((num,4))#初始化元素为0的，行号数个列表，其中每个元素仍是列表，元素数是3，在此表示矩阵
    index = 0
    for i,line in enumerate(array):
        line = line.strip()#去掉一行后的回车符号
        linelist = line.split(',')#将一行根据分割符,划分成多个元素的列表
        linelist = [int(i) for i in linelist]
        returnMat[i,:] = linelist[0:4]#向矩阵赋值，注意这种赋值方式比较笨拙
    return returnMat

def path_to_lable(path):
    img_path = path+'/img/0001.jpg'
    img = misc.imread(img_path,mode='RGB')
    w,h,_ = img.shape
    print('image_size:',w,h)
    path = path+'/groundtruth_rect.txt'
    data = file2list(path)
    size = np.array([w,h,w,h])
    data = np.divide(data,size)
    pre_lable = np.delete(data,0,axis=0)
    now_lable = np.delete(data,-1,axis=0)
    print(pre_lable.shape,now_lable.shape)
    return pre_lable,now_lable

def main():
    dir = 'F:/object_track/data/'
    file_dir = os.listdir(dir)
    pre_data = []
    now_data = []
    pre_lable = []
    now_lable = []
    for i,name in enumerate(file_dir):
        path = dir+name
        print(path)
        data1,data2 = path_to_image(path)
        y1,y2 = path_to_lable(path)
        pre_data.append(data1)
        now_data.append(data2)
        pre_lable.append(y1)
        now_lable.append(y2)

    pre_data = np.concatenate(pre_data,axis=0)
    now_data = np.concatenate(now_data,axis=0)
    pre_lable = np.concatenate(pre_lable,axis=0)
    now_lable = np.concatenate(now_lable,axis=0)

    np.save('npy_data/pre.npy',pre_data)
    np.save('npy_data/now.npy',now_data)
    np.save('npy_data/pre_lable.npy',pre_lable)
    np.save('npy_data/now_lable.npy',now_lable)



if __name__ == '__main__':
    main()


