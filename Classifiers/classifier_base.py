'''
Classifier base class
Implements functions init, fit, predict and evaluate
'''
class Classifier_base:

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
	def __init__(self, params):
		pass


	'''
	Train classifier with training data.
	All required parameters for training except for data is passed to the 
	constructor. 
	@param train_generator: Generator for training data
	@param validation_generator: Generator for validation data
	@param steps: Number of batches per epoch
	'''
	def fit(self, train_generator, validation_generator, steps):
		pass


	'''
	Predict class labels on test data
	@param test_imgs: test data
	@return predictions for test_imgs
	'''
	def predict(self, test_imgs):
		pass


	'''
	Evaluate classifier performance of validation data
	@param validation_generator: Generator for validation data
	@param steps: Number of batches per epoch
	@return classification loss
	'''
	def evaluate(self, validation_generator, steps):
		pass