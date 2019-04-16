import sys
sys.path.append('../')
from data_construct import travel_files, data_set_split
from data_generator import *
from unet2d_model import *
from utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import os
import time
from keras.callbacks import TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')

def main():
    file_path = 'E:/DATA/DCMS/'
    file_items = travel_files(file_path)
    partition = data_set_split(file_items)

    params = {'dim': (256, 256),
              'batch_size': 5,
              'n_channels': 1,
              'shuffle': True}

    test_params = {'dim': params['dim'],
                   'batch_size': params['batch_size'],
                   'n_channels': params['n_channels'],
                   'shuffle': False}

    predict_params = {'dim': params['dim'],
                      'percent': 0.1,
                      'batch_size': params['batch_size'],
                      'n_channels': params['n_channels'],
                      'save_path': 'test_result/'}

    net_conf = {'pretrained_weights': None,
                'input_size': (256, 256, 1),
                'depth': 4,
                'n_base_filters': 64,
                'optimizer': Adam,
                'activation': LeakyReLU,
                'batch_normalization': True,
                'initial_learning_rate': 5e-4,
                'loss_function': dice_coefficient_loss,
                'multi_gpu_num': 0}

    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['test'], **params)
    test_generator = DataGenerator(partition['holdout'], **test_params)
    predict_generator = predictGenerator(partition['holdout'], **predict_params)

    early_stoping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint(filepath='train_result/weights/unet_bn_t_2d-{epoch:02d}-{val_acc:.5f}.hdf5',
                                       monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                       mode='auto', period=1)
    model_name = "unet_bn_t_2d-" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

    model = unet_bn_t(**net_conf)
    results = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=50,
                                  callbacks=[model_checkpoint, early_stoping, tensorboard])

    p_test = model.predict_generator(predict_generator, steps=int(np.ceil(len(partition['holdout']) * predict_params['percent'])), verbose=1)
    saveResult(predict_params['save_path'], p_test)
    eva = model.evaluate_generator(test_generator, verbose=1)
    print(">> Testing dataset accuracy = {:.2f}%".format(eva[1] * 100.0))
    print(">> Testing dataset binary_crossentropy = {:.2f}".format(eva[2] * 1.0))
    print(">> Testing dataset mIoU  = {:.2f}%".format(eva[3] * 100.0))
    print(">> Testing dataset mDice = {:.2f}%".format(eva[4] * 100.0))
    print('Test_Accuracy: ', np.mean(results.history['val_acc']))
    plothistory(results, model_name)


if __name__ == '__main__':
    main()


