import matplotlib.pyplot as plt
import os
import skimage.io as io
import numpy as np

def plothistory(history, model_name):

    fig_accuracy = plt.figure()
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_accuracy.savefig('train_result/Model accuracy' + model_name + '.png')

    fig_loss = plt.figure()
    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig_loss.savefig('train_result/Model loss' + model_name + '.png')

def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        imgdata = img * 255
        io.imsave(os.path.join(save_path, str(i) + "_predict.png"), imgdata.clip(0, 255, imgdata).astype(np.uint8))
