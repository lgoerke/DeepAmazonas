import numpy as np
import pandas as pd
import tifffile 
import cv2
import os

### STILL NEED TO IMPORT A CLASSIFIER WITH THE METHODS fit(x,y) and predict(X) FOR THIS SCRIPT TO WORK

LABELS = {'blow_down': 0,
          'bare_ground': 1,
          'conventional_mine': 2,
          'blooming': 3,
          'cultivation': 4,
          'artisinal_mine': 5,
          'haze': 6,
          'primary': 7,
          'slash_burn': 8,
          'habitation': 9,
          'clear': 10,
          'road': 11,
          'selective_logging': 11,
          'partly_cloudy': 12,
          'agriculture': 13,
          'water': 14,
          'cloudy': 15}
          

 
def test_Nodes():
    '''
     |       Little test tree... 
     |
    / \
    
    '''
    node_2_2_forestphenomena = Node([],['blow_down','blooming','slash_burn','selective_logging'])
    node_2_1_infrstructure = Node([],['conventional_mine','artisinal_mine','road'])
    node_1_land = Node([node_2_1_infrstructure, node_2_2_forestphenomena ],['agriculture','habitation','water','primary','cultivation','bare_ground'],['cloudy'])
    node_0_weather = Node([node_1_land], ['cloudy','haze','partly_cloudy','clear'])
    return node_0_weather
    
def load_example(n = 200, path = '../input/train_tif_v2/' ):
    '''
    load a few pictures/targets info a sample matrix.
    '''
        
    train_path = '/media/sebastian/7B4861FD6D0F6AA2/train-tif-v2/'
    test_path = 'input/test-jpg/'
    train_labels = pd.read_csv(path+'input/train.csv')
    #test_labels = pd.read_csv('input/sample_submission.csv')
        
    X = np.zeros((n,224,224,4))
    y = np.zeros((n,17))

    
    for i, imgstr in enumerate(train_labels.image_name.values):
        p = train_path+imgstr+'.tif'
        
        if i>=n: return X,y
        
        img = cv2.resize(tifffile.imread(p),(224,224))
        X[i,:,:,:] = img # cv2.resize(img,(255,255))
        
        
        i=+i
        

        for t in train_labels.tags.values[i].split(' '):
            y[i,LABELS[t]] = 1

    
class Node():
    '''One node in a decision tree, which creates one clsfr instance and can create a dataset for it.'''
    
    def __init__(self,children, use_labels, exclude_classes = [], weights = None, clsfr=None):
        '''
        @args:
        children: array of Node()s that receive their input from this Node.
        use_lables: Array of labels to include ['forest','clouds']
        exclude_classes: Labels to not use when classified as 1 already.
        
        '''
        self.X_mask = None
        self.y_history = None
        self.use_lables = use_labels
        self.lbls = None
        self.children = children
        self.exclude_classes = exclude_classes
        if clsfr is None:
            self.clsfr = self.__init_clsfr__()
        else:
            self.clsfr = clsfr
         
    def __init_clsfr__(self, size = None):
        '''Create a new classifier object bound to this node'''
        
        return SimpleNet(n_classes = len(self.use_lables), nb_epoch = 1, batch_size= 4)
    
    def create_training_set(self, X,t, y_history):
        '''create a training set tailored to this decision operation.
        
        @args:
        X: Pointer to your image set
        y: target values
        y_history: np array documenting the current classification situation: 

        '''
        
        t = np.copy(t)
        
        #Cut out unwanted objects that have been classified as sth before (e.g. all cloudy > .5)        
        X_mask = np.ones((X.shape[0]))
        
        for i, cat in enumerate(self.exclude_classes):
            X_mask = np.multiply(X_mask,(y_history[:,LABELS[cat]] < 0.4))
            
        X1 = np.copy(X[(X_mask.astype(np.bool))])
            
        #Cut away all uninportant class labels from y:
        lbls = np.zeros((len(LABELS.keys())))
        for lbl in (self.use_lables):
            lbls[LABELS[lbl]] = True      #TODO: make 1-dimensional
        
        t1 = t[:,lbls.astype(np.bool)]
        t1 = t1[(X_mask.astype(np.bool))]
        
        self.y_history = y_history
        self.X_mask = X_mask
        self.lbls = lbls
        
        return X1,t1
        
    def apply(self, X, y_history = None):
        '''
        Apply this node's classifier on X using the prior information y_history 
        from earlier classifiers.
        
        @args:
        -X classifier input
        -y_history: prior information from earlier clsfrs.
        
        @return:
        y_history to use as prior input for further classification
        
        '''
        
        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((X.shape[0],len(self.lbls)))
        
        X_mask = np.ones((X.shape[0]))
        for i, cat in enumerate(self.exclude_classes):
            X_mask = np.multiply(X_mask,(y_history[:,LABELS[cat]] <0.4))
        
        X1 = np.copy(X[X_mask.astype(np.bool)])
        
        print(X1.shape)
        
        dy = self.clsfr.predict(X1)
        
        for n,i in enumerate(np.where(X_mask)[0]):
            y_history[i,:][self.lbls.astype(np.bool)[:]]=y_history[i,:][self.lbls.astype(np.bool)[:]]+dy[n,:] #TODO: fix update rule!
            
        return y_history
        
    def apply_rec(self, X, y_history = None):
        '''
        Classify images recursively in the Node() tree
        '''
        
        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((X.shape[0],self.lbls.shape[0]))
        
        y_history = self.apply(X, y_history)
            
        for child in self.children:
            
            y_history = child.apply_rec(X, y_history)
                
        return y_history
        
        
    def train(self, X,t ,y_history = None):
        '''
        Train this node's classifier
        
        @args:
        X: X training data
        t: training targets
        y_history: prior training data
        
        '''
        
        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((t.shape))
                    
        X1,y1 = self.create_training_set(X,t,y_history)
            
        self.clsfr.fit(X1,y1)
        
    def train_rec(self, X,t,y_history = None):
        '''
        Train this clsfr and its children recursively
        '''
        
        if y_history is None:
            print('Created new y-matrix')
            y_history = np.zeros((t.shape))

        self.train(X,t,y_history)
        
        y_history = self.apply(X,y_history)
        
        for child in self.children:
            child.train_rec(X,y,y_history)
            
