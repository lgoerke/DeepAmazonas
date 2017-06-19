from multiprocessing import Pool, cpu_count
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import fbeta_score
from PIL import Image, ImageStat
import random
from sklearn.utils import shuffle
from skimage import io
import pandas as pd
import numpy as np
import glob, cv2
import random
import scipy

random.seed(1)
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')

def get_features(path):
    try:
        st = []
        #pillow jpg
        img = Image.open(path)
        im_stats_ = ImageStat.Stat(img)
        st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
        img = np.array(img)[:,:,:3]
        st += [scipy.stats.kurtosis(img[:,:,0].ravel())]
        st += [scipy.stats.kurtosis(img[:,:,1].ravel())]
        st += [scipy.stats.kurtosis(img[:,:,2].ravel())]
        st += [scipy.stats.skew(img[:,:,0].ravel())]
        st += [scipy.stats.skew(img[:,:,1].ravel())]
        st += [scipy.stats.skew(img[:,:,2].ravel())]
        #cv2 jpg
        img = cv2.imread(path)
        bw = cv2.imread(path,0)
        st += list(cv2.calcHist([bw],[0],None,[256],[0,256]).flatten()) #bw 
        st += list(cv2.calcHist([img],[0],None,[256],[0,256]).flatten()) #r
        st += list(cv2.calcHist([img],[1],None,[256],[0,256]).flatten()) #g
        st += list(cv2.calcHist([img],[2],None,[256],[0,256]).flatten()) #b
        m, s = cv2.meanStdDev(img) #mean and standard deviation
        st += list(m)
        st += list(s)
        st += [cv2.Laplacian(bw, cv2.CV_64F).var()] 
        st += [cv2.Laplacian(img, cv2.CV_64F).var()]
        st += [cv2.Sobel(bw,cv2.CV_64F,1,0,ksize=5).var()]
        st += [cv2.Sobel(bw,cv2.CV_64F,0,1,ksize=5).var()]
        st += [cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5).var()]
        st += [cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5).var()]
        st += [(bw<30).sum()]
        st += [(bw>225).sum()]
    except:
        print(path)
    return [path, st]

def normalize_img(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_features, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    return fdata

in_path = './input/'
train = pd.read_csv(in_path + 'train_v2.csv')
train['path'] = train['image_name'].map(lambda x: in_path + 'train-jpg/' + x + '.jpg')
y = train['tags'].str.get_dummies(sep=' ')
xtrain = normalize_img(train['path']); print('train...')

weather_conditions = y.loc[:, ['clear', 'partly_cloudy', 'cloudy', 'haze']]
#transform dataframe to a list to shuffle
list_weath_cond = weather_conditions.values.tolist()
#unison shuffle
s_xtrain, s_list_weath_cond = shuffle(xtrain, list_weath_cond)
#from list back to dataframe
s_weath_cond = pd.DataFrame(s_list_weath_cond, columns=['clear', 'partly_cloudy', 'cloudy', 'haze'])

sample_tr = s_xtrain[:30000]
sample_val = s_xtrain[30000:]

print(len(sample_tr))
print(len(sample_val))

etr = ExtraTreesRegressor(n_estimators=200, max_depth=30, n_jobs=-1, random_state=1)
etr.fit(sample_tr, s_weath_cond[:30000]); print('etr fit...')

val_pred = etr.predict(sample_val)
val_pred[val_pred > 0.2] = 1
val_pred[val_pred < 1] = 0
print(fbeta_score(s_weath_cond[30000:],val_pred,beta=2, average='samples'))

test_jpg = glob.glob(in_path + 'test-jpg/*')
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg])
test.columns = ['image_name','path']
xtest = normalize_img(test['path']); print('test...')

test_pred = etr.predict(xtest)
test_pred[test_pred > 0.2] = 1
test_pred[test_pred < 1] = 0



