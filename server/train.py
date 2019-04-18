import sys
sys.path.append('../')
from data_construct import travel_files, data_set_split
from data_generator import *
from unet2d_model import *
from utils import *
from configure import GetConfigure
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import os
import time
import warnings
warnings.filterwarnings('ignore')


def main():
    # ## load configure
    file_path, epochs, chosen_file_percent, params, test_params, predicting_params, net_conf, cudas, model_type = GetConfigure()
    os.environ["CUDA_VISIBLE_DEVICES"] = cudas
    # ## configure dataset
    file_items = travel_files(file_path)
    partition = data_set_split(file_items, chosen_file_percent)
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validate'], **params)
    test_generator = DataGenerator(partition['test'], **test_params)
    predicting_generator = predictGenerator(partition['test'], **predicting_params)

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
    p_test = model.predict_generator(predicting_generator, steps=int(np.ceil(len(partition['test']) * predicting_params['percent'])), verbose=1)
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


