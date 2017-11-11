from scipy import misc
import numpy as np
import os

dir = 'F:/object_track/data/Dancer2/img/'

def to_npy(data):
    m = len(data)
    npdata = np.zeros([m,224,224,3])
    for i,name in enumerate(data):
        name = dir+name
        temp = misc.imread(name,mode='RGB')
        temp = misc.imresize(temp,[224,224])
        temp = np.expand_dims(temp,axis=0)
        npdata[i] = temp
    return npdata


def main():
    file_names = os.listdir(dir)
    pre_names = file_names.copy()
    del pre_names[-1]

    now_names = file_names.copy()
    del now_names[0]

    print(len(pre_names))
    print(len(now_names))

    data1 = to_npy(pre_names)
    data2 = to_npy(now_names)

    print(data1.shape)
    print(data2.shape)
    np.save('npy_data/data1.npy',data1)
    np.save('npy_data/data2.npy', data2)




if __name__ == '__main__':
    main()