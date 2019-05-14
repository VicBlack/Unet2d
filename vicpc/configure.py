import sys
sys.path.append('../')
from data_construct import *
from data_generator import *
from unet2d_model import *
from utils import *
from keras.optimizers import *
from keras.layers import *
from keras.preprocessing.image import *

# ## training configure
file_path = 'E:/DATA/DCMS/'
epochs = 3
chosen_file_percent = 0.001
predict_percent = 0.1
params = {'dim': (256, 256),
          'batch_size': 2,
          'n_channels': 1}

net_conf = {'pretrained_weights': None,
            'input_size': (256, 256, 1),
            'depth': 2,
            'n_base_filters': 32,
            'optimizer': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
            'activation': ReLU,
            'batch_normalization': True,
            'loss_function': dice_coefficient_loss,
            'dropout': 0.3,
            'multi_gpu_num': 0}

cudas = "0"

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
    dtype=np.float64)

# ## configure net
# ## available net below:
# 'unet_2d'
# 'unet_bn_upsampling_2d'
# 'unet_bn_deconv_2d'
# 'unet_bn_full_upsampling_dp_2d'
# 'unet_bn_full_deconv_dp_2d'
# 'unet_bn_deconv_upsampling_dp_2d'
# 'unet_bn_upsampling_deconv_dp_2d'
# 'unet_dense_2d'
# 'unet_bn_block_full_upsampling_dp_2d'
model_type = 'unet_gn_upsampling_2d'


def GetConfigure():
    return file_path, epochs, chosen_file_percent, predict_percent, params, net_conf, cudas, model_type, datagen
