import matplotlib.pyplot as plt
import os
import skimage.io as io
import numpy as np
import sys
import json

def plothistory(history, figure_path):
    # 绘制训练 & 验证的准确率值
    for metrictype in history:
        if 'val_' == metrictype[0:4]:
            continue
        fig = plt.figure()
        plt.plot(history[metrictype], label='train ' + metrictype)
        plt.plot(history['val_' + metrictype], label='val ' + metrictype)
        plt.title('Model ' + metrictype)
        plt.ylabel(metrictype)
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(loc='upper left')
        fig.savefig(os.path.join(figure_path, metrictype + '.png'), dpi=300)


def plot_acc_loss(history, figure_path):
    fig = plt.figure()
    # 绘制训练 & 验证的acc & loss值
    plt.plot(history['acc'], 'r', label='train acc')
    # loss
    plt.plot(history['loss'], 'g', label='train loss')
    # val_acc
    plt.plot(history['val_acc'], 'b', label='val acc')
    # val_loss
    plt.plot(history['val_loss'], 'k', label='val loss')
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
    def __init__(self, filename='train.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__=='__main__':
    conf_path = 'E:/WorkSpace/PYSpace/Heart/Unet2d/vicpc/train_result/configures/unet_gn_upsampling_2d_B2_SGD_ReLU_drop0.3_da-20190522-165901'
    figure_path = 'E:/WorkSpace/PYSpace/Heart/Unet2d/vicpc/train_result/figures/unet_gn_upsampling_2d_B2_SGD_ReLU_drop0.3_da-20190522-165901'
    with open(os.path.join(conf_path, "history.json"), "r", encoding='utf-8') as f:
        history = json.load(f)
        plothistory(history, figure_path)
