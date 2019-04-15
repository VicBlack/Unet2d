import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.utils import multi_gpu_model

K.set_image_data_format('channels_last')

## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# ### Calculating metrics:
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coefficient(y_true, y_pred)



def downsampling_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', batch_normalization=False, activation=None):

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(input_tensor)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    return MaxPooling2D(pool_size=(2, 2))(x), x

def upsampling_block(input_tensor, skip_tensor, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', batch_normalization=False, activation=None):

    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)  #采用反卷积替代上采样
    x = Concatenate()([x, skip_tensor])  # 特征级联

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    return x  #返回第二次卷积的结果

def unet_bn_t_dp(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam, activation=LeakyReLU, batch_normalization=True, initial_learning_rate=5e-4, loss_function=dice_coefficient_loss, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    upsamplingtensors = []  # 用于存放上采样中，第二次卷积的结果，以供之后deep supervision使用
    for i in range(depth):
        x, x0 = downsampling_block(x, n_base_filters, batch_normalization=batch_normalization, activation=activation)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    dplayer=None

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_block(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)
        upsamplingtensors.append(x)
        if i == depth - 1:
            dplayer = upsamplingtensors[depth - i - 1]
        else:
            dplayer = Conv2DTranspose(n_base_filters, kernel_size=(2, 2), strides=(2, 2))(dplayer)  # 采用反卷积替代上采样
            dplayer = Concatenate()([upsamplingtensors[depth - i - 1], dplayer])  # 特征级联

    x = Concatenate()([x, dplayer])
    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer(lr=initial_learning_rate),
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', IoU, dice_coefficient])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_bn_t(pretrained_weights=None, input_size=(256, 256, 1), depth=4, n_base_filters=64, optimizer=Adam, activation=LeakyReLU, batch_normalization=True, initial_learning_rate=5e-4, loss_function=dice_coefficient_loss, multi_gpu_num=0):
    x = Input(input_size)
    # 输入层
    inputs = x
    skiptensors = []  # 用于存放下采样中，每个深度后的tensor，以供之后级联使用
    for i in range(depth):
        x, x0 = downsampling_block(x, n_base_filters, batch_normalization=batch_normalization, activation=activation)
        skiptensors.append(x0)
        n_base_filters *= 2
    # 最底层两次卷积操作
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)
    x = Conv2D(filters=n_base_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x) if batch_normalization else x
    x = activation()(x) if activation else Activation('relu')(x)

    for i in reversed(range(depth)):  # 下采样过程中，深度从深到浅
        n_base_filters //= 2  # 每个深度往上。特征减少一倍
        x = upsampling_block(x, skiptensors[i], n_base_filters, batch_normalization=batch_normalization, activation=activation)

    # 输出层
    outputs = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    if multi_gpu_num:
        model = multi_gpu_model(model, gpus=multi_gpu_num)

    model.compile(optimizer=optimizer(lr=initial_learning_rate),
                  loss=loss_function,
                  metrics=['accuracy', 'binary_crossentropy', IoU, dice_coefficient])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coefficient_loss, metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__=='__main__':
    m = unet_bn_t_dp()
