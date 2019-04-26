import matplotlib.pyplot as plt
import os
import skimage.io as io
import numpy as np
import sys


def plothistory(history, figure_path, type):
    fig = plt.figure()
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history[type], label='train ' + type)
    plt.plot(history.history['val_' + type], label='val acc ' + type)
    plt.title('Model ' + type)
    plt.ylabel(type)
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='upper left')
    fig.savefig(os.path.join(figure_path, type + '.png'), dpi=300)


def plot_acc_loss(history, figure_path):
    fig = plt.figure()
    # 绘制训练 & 验证的acc & loss值
    plt.plot(history.history['acc'], 'r', label='train acc')
    # loss
    plt.plot(history.history['loss'], 'g', label='train loss')
    # val_acc
    plt.plot(history.history['val_acc'], 'b', label='val acc')
    # val_loss
    plt.plot(history.history['val_loss'], 'k', label='val loss')
    plt.title('Model Acc Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc='upper left')
    fig.savefig(os.path.join(figure_path, 'model acc_loss.png'), dpi=300)


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        imgdata = img * 255
        io.imsave(os.path.join(save_path, str(i) + "_predict.png"), imgdata.clip(0, 255, imgdata).astype(np.uint8))


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
