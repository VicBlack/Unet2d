import sys
sys.path.append('../')
from data_construct import travel_files, data_set_split
from data_generator import *
from unet2d_model import *
from utils import *
from keras.optimizers import Adam
from keras.layers import LeakyReLU

# ## training configure
file_path = 'E:/WorkSpace/CAP/DCMS/'
epochs = 50
chosen_file_percent = 1.0
params = {'dim': (256, 256),
          'batch_size': 4,
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

cudas = "0"

# ## configure net
# ## available net below:
# 'unet_2d'
# 'unet_bn_upsampling_2d'
# 'unet_bn_deconv_2d'
# 'unet_bn_full_upsampling_dp_2d'
# 'unet_bn_full_deconv_dp_2d'
# 'unet_bn_deconv_upsampling_dp_2d'
# 'unet_bn_upsampling_deconv_dp_2d'
model_type = 'unet_bn_full_upsampling_dp_2d'


def GetConfigure():
    return file_path, epochs, chosen_file_percent, params, test_params, predicting_params, net_conf, cudas, model_type
