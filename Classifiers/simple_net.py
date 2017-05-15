from Classifiers.classifier_base import Classifier_base


from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from keras.layers.advanced_activations import PReLU

'''
Simple neural network classifier
'''
class SimpleNet(Classifier_base):

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
	def __init__(self, shape=(224, 224), n_classes=2, nb_epoch = 12, lr=0.001, batch_size=64, optimizer='adam'):
		
		self.shape = shape
        	self.n_classes = n_classes
        	self.nb_epoch = nb_epoch
        	self.lr = lr
        	self.optimizer = optimizer
        	self.batch_size = batch_size

		self.build()


	def make_block(self, out_channels, kernel_size=3, stride=1, padding=1):

		kernel_initializer = "he_normal"
		kernel_regularizer = l2(0.0001)

		def f(input):

	      	conv = Conv2D(filters=out_channels, kernel_size=kernel_size,
	          	strides=stride, padding=padding,
	          	kernel_initializer=kernel_initializer,
				kernel_regularizer=kernel_regularizer)(input)
			batch = BatchNormalization(axis=3)(conv)
			activation = PReLU()(batch)

			return activation

		return f	


	def build(self):

		kernel_initializer="he_normal"

		x = Input(shape=self.shape)

		x = make_block(8, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(8, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(8, kernel_size=1, stride=1, padding=0)(x)

		x = make_block(32, kernel_size=3, stride=1, padding=1)(x)
		x = make_block(32, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(32, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(32, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(32, kernel_size=3, stride=1, padding=1)(x)
		x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
		

		x = make_block(64, kernel_size=3, stride=1, padding=1)(x)
		x = make_block(64, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(64, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(64, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(64, kernel_size=3, stride=1, padding=1)(x)
		x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
		x = Dropout(0.25)(x)

		outa = AveragePooling2D(x.output_shape)(x)
		outa = Flatten(outa)

		x = make_block(128, kernel_size=3, stride=1, padding=1)(x)
		x = make_block(128, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(128, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(128, kernel_size=1, stride=1, padding=0)(x)
		x = make_block(128, kernel_size=3, stride=1, padding=1)(x)
		x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
		x = Dropout(0.50)(x)

		outb = AveragePooling2D(x.output_shape)(x)
		outb = Flatten(outb)

		out = Concatenate([outa, outb], axis=3)

		x = Dense(512,kernel_initializer="he_normal")(out)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)

		x = Dense(512,kernel_initializer="he_normal")(x)
		x = BatchNormalization(axis=3)(x)
		x = Activation('relu')(x)


		if self.optimizer == 'adam':
            opt = optimizers.adam(lr=self.lr)
        elif self.optimizer == 'adadelta':
            opt = optimizers.adadelta()
        else:
            opt=optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0005, nesterov=True)

		self.model = Dense(self.n_classes, activation='sigmoid')(x)

		self.model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=["accuracy"])

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
        	callbacks=[early], epochs = self.nb_epoch, verbose = 1)




	'''
	Predict class labels on test data
	@param test_imgs: test data
	@return predictions for test_imgs
	'''
	def predict(test_imgs):
		return self.model.predict(X_test, batch_size = self.batch_size, verbose = 1)


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
