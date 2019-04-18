import numpy as np
import keras
import skimage.io as io
import os
import random
from data_preprocess import img_load, lab_load, img_crop, img_norm

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=2, dim=(256, 256), n_channels=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        ## 构造一个[0,....,sampls]得列表
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.list_IDs)/float(self.batch_size)))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        ## 根据索引找到每条索引对应得文件名
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        ##根据文件名加载数据
        X, y1 = self.__data_generation(list_IDs_temp)
        return X, y1

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size,  *self.dim, self.n_channels))
        # print('X.shape', X.shape)
        y1 = np.empty((self.batch_size, *self.dim, 1))
        # print('y1.shape', y1.shape)
        for i, item in enumerate(list_IDs_temp):
            img_array = img_load(item[0], shape=(256, 256), norm=True)
            lab_array = lab_load(item[1], shape=(256, 256), norm=False, binary=True)
            img_array = img_array.reshape(*img_array.shape, self.n_channels)
            lab_array = lab_array.reshape(*lab_array.shape, self.n_channels)
            # print('each_img_shape', img_array.shape)
            # print('each_lab_shape', lab_array.shape)
            X[i,] = img_array
            y1[i,] = lab_array
        return X, y1

    def load_data(self):
        for i in range(self.__len__()):
            X, y1 = self.__getitem__(i)
            yield X

# class predict_DataGenerator(DataGenerator):
#
#     def __init__(self, list_IDs, batch_size=2, dim=(256, 256), n_channels=1, shuffle=False, save_path='test_result/'):
#         DataGenerator.__init__(self, list_IDs, batch_size=batch_size, dim=dim, n_channels=n_channels, shuffle=shuffle)
#         self.save_path = save_path
#
#     def __getitem__(self, index):
#         indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
#         ## 根据索引找到每条索引对应得文件名
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#         ##根据文件名加载数据
#         X = self.__data_generation(list_IDs_temp)
#         return X
#
#     def __data_generation(self, list_IDs_temp):
#         X = np.empty((self.batch_size,  *self.dim, self.n_channels))
#         for i, item in enumerate(list_IDs_temp):
#             img_array = img_load(item[0], shape=(256, 256), norm=True)
#             save_name = str(i) + "_source_" + os.path.splitext(os.path.split(item[0])[1])[0] + '.png'
#             io.imsave(os.path.join(self.save_path, save_name), img_array)
#             img_array = img_array.reshape(*img_array.shape, self.n_channels)
#             X[i, ] = img_array
#         return X
#

def predictGenerator(test_path, batch_size=2, percent=1, dim=(256, 256), n_channels=1, save_path='test_result/'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files = random.sample(test_path, int(np.ceil(len(test_path) * percent)))
    for i, item in enumerate(files):
        img_array = img_load(os.path.join(item[0]), shape=dim, norm=True)
        save_name = str(i) + "_source_" + os.path.splitext(os.path.split(item[0])[1])[0] + '.png'
        imgdata = img_array * 255
        io.imsave(os.path.join(save_path, save_name), imgdata.clip(0, 255, imgdata).astype(np.uint8))
        img_array = np.reshape(img_array, img_array.shape + (1,))
        img_array = np.reshape(img_array, (1,) + img_array.shape)
        yield img_array







