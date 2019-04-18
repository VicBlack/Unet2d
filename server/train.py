import sys
sys.path.append('../')
from data_construct import travel_files, data_set_split
from data_generator import *
from unet2d_model import *
from utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')


def main():
    # ## training configure
    file_path = '/data/data/DCMS/'
    epochs = 50
    params = {'dim': (256, 256),
              'batch_size': 12,
              'n_channels': 1,
              'shuffle': True}

    test_params = {'dim': params['dim'],
                   'batch_size': params['batch_size'],
                   'n_channels': params['n_channels'],
                   'shuffle': False}

    predicting_params = {'dim': params['dim'],
                      'percent': 0.01,
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

    # ## configure dataset
    file_items = travel_files(file_path)
    partition = data_set_split(file_items)
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validate'], **params)
    test_generator = DataGenerator(partition['test'], **test_params)
    predicting_generator = predictGenerator(partition['test'], **predicting_params)

    # ## configure net
    # ## available net below:
    # 'unet_2d'
    # 'unet_bn_upsampling_2d'
    # 'unet_bn_deconv_2d'
    # 'unet_bn_full_upsampling_dp_2d'
    # 'unet_bn_deconv_dp_2d'
    model_type = 'unet_bn_upsampling_2d'
    model = GetNet(model_type, net_conf)
    early_stoping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint(filepath='train_result/weights/' +
                                                model_type + '-{epoch:02d}-{val_loss:.5f}.hdf5',
                                       monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                       mode='auto', period=1)
    model_name = model_type + '-' + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

    # ## training on train_dataset and validate_dataset
    print('>> Start Training')
    results = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs,
                                  callbacks=[model_checkpoint, early_stoping, tensorboard])

    # ## predict on test_dataset
    print('>> Start Predicting')
    p_test = model.predict_generator(predicting_generator, steps=int(np.ceil(len(partition['holdout']) * predicting_params['percent'])), verbose=1)
    saveResult(predicting_params['save_path'], p_test)

    # ## evaluate on test_dataset
    print('>> Start Evaluating')
    eva = model.evaluate_generator(test_generator, verbose=1)
    print(">> Testing dataset accuracy = {:.3f}%".format(eva[1] * 100.0))
    print(">> Testing dataset binary_crossentropy = {:.3f}".format(eva[2] * 1.0))
    print(">> Testing dataset mIoU  = {:.3f}%".format(eva[3] * 100.0))
    print(">> Testing dataset mDice = {:.3f}%".format(eva[4] * 100.0))
    print('Test_Accuracy: ', np.mean(results.history['val_acc']))
    plothistory(results, model_name)


if __name__ == '__main__':
    main()
