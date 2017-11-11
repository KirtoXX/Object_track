import numpy as np
import scipy.io as sio

path = "F:/object_track/data/Dancer2/groundtruth_rect.txt"

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
        print(linelist)
        returnMat[i,:] = linelist[0:4]#向矩阵赋值，注意这种赋值方式比较笨拙
    return returnMat

def main():
    data = file2list(path)
    size = np.array([320,262,320,262])
    data = np.divide(data,size)
    print(data[1])
    print(data[2])
    y1 = np.delete(data,0,axis=0)
    y2 = np.delete(data,-1,axis=0)

    print(y1.shape)
    np.save('npy_data/y1.npy',y1)
    np.save('npy_data/y2.npy',y2)


if __name__ == '__main__':
    main()