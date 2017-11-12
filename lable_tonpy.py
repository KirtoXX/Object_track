import numpy as np
import scipy.io as sio
from scipy import misc
import os

path1 = "/vot2016"

def file2list(filename):
    fr = open(filename)
    array = fr.readlines() #以文件中的每行为一个元素，形成一个list列表
    num = len(array)
    returnMat = np.zeros((num,8))#初始化元素为0的，行号数个列表，其中每个元素仍是列表，元素数是3，在此表示矩阵
    index = 0
    for i,line in enumerate(array):
        line = line.strip()#去掉一行后的回车符号
        linelist = line.split(',')#将一行根据分割符,划分成多个元素的列表
        #linelist = [int(i) for i in linelist]
        #print(linelist)
        returnMat[i,:] = linelist[0:8]#向矩阵赋值，注意这种赋值方式比较笨拙
    return returnMat

def path_to_lable(path):
    img_path = path+'/00000001.jpg'
    img = misc.imread(img_path,mode='RGB')
    w,h,_ = img.shape
    print('image_size:',w,h)
    path = path+'/groundtruth.txt'
    data = file2list(path)
    #divid lable------
    size = np.array([w,h,w,h,w,h,w,h])
    data = np.divide(data,size)
    #del the -1,and 0 lable
    pre_lable = np.delete(data,-1,axis=0)
    now_lable = np.delete(data,0,axis=0)
    print(pre_lable.shape,now_lable.shape)
    return pre_lable,now_lable

def main():
    dir = 'vot2016'
    file_dir = os.listdir(dir)
    pre_lable = []
    now_lable = []
    for i, name in enumerate(file_dir):
        path = dir+"/"+name
        print(path)
        y1, y2 = path_to_lable(path)
        pre_lable.append(y1)
        now_lable.append(y2)

    pre_lable = np.concatenate(pre_lable, axis=0)
    now_lable = np.concatenate(now_lable, axis=0)
    pre_lable = pre_lable.astype(dtype=np.float32)
    now_lable = now_lable.astype(dtype=np.float32)
    print(pre_lable.shape,now_lable.shape)
    np.save('npy_data/pre_lable.npy', pre_lable)
    np.save('npy_data/now_lable.npy', now_lable)


if __name__ == '__main__':
    main()