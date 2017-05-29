from Classifiers.classifier_base import Classifier_base

from keras import applications
from keras.layers import Input, Activation
from keras.layers import AveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras import optimizers

class VGG16(Classifier_base):

    '''
    Class variable containing parameters to optimize
    together with ranges.
    E.g. hp.uniform( 'dropout', 0, 0.6)
    '''
    space = (

    )

    '''
    Constructor
    @params: list of all model parameters
    '''
    def __init__(self, shape=(256, 256, 3), n_classes=2, nb_epoch = 12, lr=0.001, batch_size=64, optimizer='adam', nl_freeze=15):
        
        self.shape = shape
        self.n_classes = n_classes
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.nl_unfreeze = nl_freeze

        self.build()

    def build(self):
	"""
        Loads preconstructed VGG model from keras without top classification layer;
        Stacks custom classification layer on top;
        Returns stacked model
        """

        # build the VGG16 network
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=shape)
        print('Base-model loaded.')

	for layer in base_model.layers:
            layer.trainable = False

	top_model = base_model.output
        top_model = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(top_model)
        top_model = Flatten(name='flatten')(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dropout(0.5)(top_model)
        top_model = Dense(self.n_classes, activation='sigmoid', name='predictions')(top_model)
        
        model = self.model = Model(base_model.input, top_model)

	if self.optimizer == 'adam':
            opt = optimizers.adam(lr=self.lr)
        elif self.optimizer == 'adadelta':
            opt = optimizers.adadelta(lr = self.lr)
        else:
            opt=optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=True)
        
        model.compile(loss='binary_crossentropy', 
                      optimizer=opt,
                      metrics=['accuracy'])
        
        return model
   
    def unfreeze_layers(self)
	for layer in self.model.layers[:self.nl_freeze]:
                layer.trainable = False
        for layer in self.model.layers[self.nl_freeze:]:
                layer.trainable = True

        self.model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
        
        
    def fit(self, train_generator, validation_generator, steps):
        
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

        self.model.fit_generator(train_generator, steps_per_epoch=steps[0]//self.batch_size, 
            validation_data=validation_generator, validation_steps=steps[1]//self.batch_size,
            callbacks=[early], epochs = self.nb_epoch, verbose = 1)
        
    '''
    Predict class labels on test data
    @param test_imgs: test data
    @return predictions for test_imgs
    '''
    def predict(self, Xtest):
        return self.model.predict(X_test, batch_size = self.batch_size, verbose = 1)
