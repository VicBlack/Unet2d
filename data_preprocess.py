import pydicom
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt

def img_crop(data, cropshape=(256, 256)):
    cropwidth = cropshape[0]
    cropheight = cropshape[1]
    data = np.array(data)
    croped = np.zeros((cropwidth, cropheight), dtype=np.uint16)
    wpos = int(abs((cropwidth - data.shape[0])/2))
    hpos = int(abs((cropheight - data.shape[1])/2))
    if(data.shape[0] < cropwidth):
        if(data.shape[1] < cropheight):
            croped[wpos: wpos + data.shape[0], hpos: hpos + data.shape[1]] = data
        else:
            croped[wpos: wpos + data.shape[0], :] = data[:, hpos: hpos + cropheight]
    else:
        if (data.shape[1] < cropheight):
            croped[:, hpos: hpos + data.shape[1]] = data[wpos:wpos + cropwidth, :]
        else:
            croped = data[wpos:wpos + cropwidth, hpos: hpos + cropheight]
    return croped

def img_load(item, shape=None, norm=True):
    img = pydicom.dcmread(item).pixel_array
    if(shape):
        img = img_crop(img, shape)
    if(norm):
        img = img_norm(img)
    return img

def lab_load(item, shape=None, norm=False, binary=True):
    lab = io.imread(item, as_gray=True)
    if (shape):
        lab = img_crop(lab, shape)
    if (norm):
        lab = img_norm(lab)
    if (binary):
        lab[lab.nonzero()] = 1
        # lab = lab.astype(np.bool)
    return lab

def img_norm(data):
    data = np.array(data)
    max = np.max(data)
    min = np.min(data)
    if max == min:
        return np.zeros(data.shape)
    data = (data - min)/(max - min)
    return data

if __name__=='__main__':
    file_path = 'E:/DATA/DCMS/dcm/DET0000101_SA2_ph5.dcm'
    label_path = 'E:/DATA/DCMS/masks/DET0000101_SA2_ph5.png'
    img = img_load(file_path, shape=(256, 256))
    lab = lab_load(label_path, shape=(256, 256))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(lab, 'gray')
    plt.show()




