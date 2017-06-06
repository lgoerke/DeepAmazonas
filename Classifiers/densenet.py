# -*- coding: utf-8 -*-
from Classifiers.classifier_base import Classifier_base

from keras.optimizers import SGD
from keras.layers import Input, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

from keras.callbacks import EarlyStopping
from custom_layers.scale_layer import Scale

'''
DenseNet
'''
class DenseNet(Classifier_base):

    '''
    Class variable containing parameters to optimize
    together with ranges.
    E.g. hp.uniform( 'dropout', 0, 0.6)
    '''
    space = (

    )

    '''
    Constructor
    
    DenseNet 161 Model for Keras

    Model Schema is based on 
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfVnlCMlBGTDR3RGs
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA


    @param: nb_dense_block: number of dense blocks to add to end
    @param: growth_rate: number of filters to add per dense block
    @param: nb_filter: initial number of filters
    @param: reduction: reduction factor of transition blocks.
    @param: dropout_rate: dropout rate
    @param: weight_decay: weight decay factor
    @param: classes: optional number of classes to classify images

    @return: A Keras model instance.
    '''
    def __init__(self, img_rows, img_cols, batch_size, nb_epoch, color_type=1, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
        
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.color_type = color_type
        self.nb_dense_block = nb_dense_block
        self.growth_rate = growth_rate
        self.nb_filter = nb_filter
        self.reduction = reduction
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        self.build()

    '''
    Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
    
    @param: x: input tensor 
    @param: stage: index for dense block
    @param: branch: layer index within each dense block
    @param: nb_filter: number of filters
    '''
    def conv_block(self, x, stage, branch, nb_filter):

        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4  
        x = BatchNormalization(epsilon=self.eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
        x = Activation('relu', name=relu_name_base+'_x1')(x)
        x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

        if self.dropout_rate:
            x = Dropout(self.dropout_rate)(x)

        # 3x3 Convolution
        x = BatchNormalization(epsilon=self.eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
        x = Activation('relu', name=relu_name_base+'_x2')(x)
        x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
        x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

        if self.dropout_rate:
            x = Dropout(self.dropout_rate)(x)

        return x

    ''' 
    Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
    @param: x: input tensor
    @param: stage: index for dense block
    @param: nb_filter: number of filters
    @param: compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
    '''
    def transition_block(self, x, stage, nb_filter, compression=1.0):

        self.eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage) 

        x = BatchNormalization(epsilon=self.eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

        if self.dropout_rate:
            x = Dropout(self.dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

        return x

    ''' 
    Build a dense_block where the output of each conv_block is fed to subsequent ones
    
    @param: x: input tensor
    @param: stage: index for dense block
    @param: nb_layers: the number of layers of conv_block to append to the model.
    @param: nb_filter: number of filters
    @param: growth_rate: growth rate
    @param: grow_nb_filters: flag to decide to allow number of filters to grow
    '''
    def dense_block(self, x, stage, nb_layers, nb_filter, grow_nb_filters=True):
    
        concat_feat = x

        for i in range(nb_layers):
            branch = i+1
            x = self.conv_block(concat_feat, stage, branch, self.growth_rate)
            concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

            if grow_nb_filters:
                nb_filter += self.growth_rate

        return concat_feat, nb_filter
 
    '''
    Build a Densenet
    '''
    def build(self):

        self.eps = 1.1e-5

        # compute compression factor
        compression = 1.0 - self.reduction

        # Handle Dimension Ordering for different backends
        global concat_axis
        if K.image_dim_ordering() == 'tf':
          concat_axis = 3
          img_input = Input(shape=(self.img_rows, self.img_cols, self.color_type), name='data')
        else:
          concat_axis = 1
          img_input = Input(shape=(self.color_type, self.img_rows, self.img_cols), name='data')

        # From architecture for ImageNet (Table 1 in the paper)
        nb_filter = self.nb_filter
        nb_layers = [6,12,36,24] # For DenseNet-161

        # Initial convolution
        x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
        x = Conv2D(self.nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
        x = BatchNormalization(epsilon=self.eps, axis=concat_axis, name='conv1_bn')(x)
        x = Scale(axis=concat_axis, name='conv1_scale')(x)
        x = Activation('relu', name='relu1')(x)
        x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        # Add dense blocks
        for block_idx in range(self.nb_dense_block - 1):
            stage = block_idx+2
            x, nb_filter = self.dense_block(x, stage, nb_layers[block_idx], nb_filter)

            # Add transition_block
            x = self.transition_block(x, stage, nb_filter, compression=compression)
            nb_filter = int(nb_filter * compression)

        final_stage = stage + 1
        x, nb_filter = self.dense_block(x, final_stage, nb_layers[-1], nb_filter)

        x = BatchNormalization(epsilon=self.eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
        x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
        x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

        x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
        x_fc = Dense(1000, name='fc6')(x_fc)
        x_fc = Activation('softmax', name='prob')(x_fc)

        model = Model(img_input, x_fc, name='densenet')

        if K.image_dim_ordering() == 'th':
          # Use pre-trained weights for Theano backend
          weights_path = 'models/densenet161_weights_th.h5'
        else:
          # Use pre-trained weights for Tensorflow backend
          weights_path = 'models/densenet161_weights_tf.h5'

        model.load_weights(weights_path, by_name=True)

        # Truncate and replace softmax layer for transfer learning
        # Cannot use model.layers.pop() since model is not of Sequential() type
        # The method below works since pre-trained weights are stored in layers but not in the model
        x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
        x_newfc = Dense(self.num_classes, name='fc6')(x_newfc)
        x_newfc = Activation('sigmoid', name='prob')(x_newfc)

        self.model = Model(img_input, x_newfc)

        # Learning rate is changed to 0.001
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        return self.model



    '''
    Train classifier with training data.
    All required parameters for training except for data is passed to the 
    constructor. 
    @param train_generator: Generator for training data
    @param validation_generator: Generator for validation data
    @param steps: Number of batches per epoch
    '''
    def fit(self, train_generator, validation_generator, steps):
        
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

        self.model.fit_generator(train_generator, steps_per_epoch=steps[0]//self.batch_size, 
            validation_data=validation_generator, validation_steps=steps[1]//self.batch_size,
            callbacks=[early], epochs = self.nb_epoch, max_q_size=30, verbose = 1)




    '''
    Predict class labels on test data
    @param test_imgs: test data
    @return predictions for test_imgs
    '''
    def predict(self, test_generator, steps):
        #return self.model.predict(test_imgs, batch_size = self.batch_size, verbose = 1)
        return self.model.predict_generator(test_generator, steps=steps, verbose=1)


    '''
    Evaluate classifier performance of validation data
    @param validation_generator: Generator for validation data
    @param steps: Number of batches per epoch
    @return classification loss
    '''
    def evaluate(self, validation_generator, steps):
        steps = steps // self.batch_size or 1
        score = self.model.evaluate_generator(validation_generator, steps=steps, workers=1)    
        return score[0] 
