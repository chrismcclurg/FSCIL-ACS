# =============================================================================
# Incremental Learning (CBCL) with Active Class Selection
#
# C McClurg, A Ayub, AR Wagner, S Rajtmajer
# =============================================================================

import numpy as np
import pickle
from classification_models.tfkeras import Classifiers
from keras.models import Model
from keras.preprocessing import image
from cv2 import resize
import pandas as pd
from PIL import Image
import tarfile
import shutil
import os

def get_data(pDataName):
    if pDataName == 'grocery':
        PATH_SRC    = './data/groceryStore/'
        PATH_TRAIN  = PATH_SRC + 'train.txt'
        PATH_TEST   = PATH_SRC + 'test.txt'
        
        train = pd.read_csv(PATH_TRAIN, sep=",", header = None)
        test = pd.read_csv(PATH_TEST, sep=",", header = None)
        
        train_labels = []
        train_images = []
        test_images = []
        test_labels = []
        
        for i in range(len(train)):
            tempPath = PATH_SRC + train.iloc[i,0]
            im = Image.open(tempPath)
            im = np.array(im)
            im = resize(im, (348,348))
            train_images.append(im)
            tempLab = train.iloc[i,1]
            train_labels.append(tempLab)
            
        for i in range(len(test)):
            tempPath = PATH_SRC + test.iloc[i,0]
            im = Image.open(tempPath)
            im = np.array(im)
            im = resize(im, (348,348))
            test_images.append(im)
            tempLab = test.iloc[i,1]
            test_labels.append(tempLab)
     
    elif pDataName == 'cifar':
        
        PATH_SRC    = './data/cifar-100-python.tar.gz'
        PATH_TRAIN  = './data/cifar-100-python/train'
        PATH_TEST   = './data/cifar-100-python/test'
        
        data = tarfile.open(PATH_SRC)
        data.extractall('./data/')
        data.close()
        
        with open(PATH_TRAIN, 'rb') as fo: train_batch = pickle.load(fo, encoding='latin1')
        with open(PATH_TEST, 'rb') as fo: test_batch = pickle.load(fo, encoding='latin1')
            
        train_images    = train_batch['data'].reshape((len(train_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        train_labels    = train_batch['fine_labels']
        test_images     = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels     = test_batch['fine_labels']
        
        shutil.rmtree('./data/cifar-100-python/')
        
    elif pDataName == 'cub':
        
        PATH_SRC    = './data/CUB_200_2011.tgz'
        PATH_TTS    = './data/CUB_200_2011/train_test_split.txt'
        PATH_IMG    = './data/CUB_200_2011/images.txt'
        PATH_LAB    = './data/CUB_200_2011/image_class_labels.txt'
        
        data = tarfile.open(PATH_SRC)
        data.extractall('./data/')
        data.close()
        
        split  = list(pd.read_csv(PATH_TTS, sep=" ", header = None)[1])
        paths = list(pd.read_csv(PATH_IMG, sep=" ", header = None)[1])
        labs = list(pd.read_csv(PATH_LAB, sep=" ", header = None)[1])
        
        train_labels = []
        train_images = []
        test_images = []
        test_labels = []
        
        for i in range(len(labs)):
            tempPath = './data/CUB_200_2011/images/' + paths[i]
            im = Image.open(tempPath)
            im = np.array(im)
            im = resize(im, (256,256))
            
            tempLab = labs[i] - 1
            tempSplit = split[i]
            
            if tempSplit == 1 and len(im.shape)>2:
                train_images.append(im)
                train_labels.append(tempLab)
            elif len(im.shape)>2:
                test_images.append(im)
                test_labels.append(tempLab)

        shutil.rmtree('./data/CUB_200_2011/')
        os.remove('./data/attributes.txt')

    else:
        print('DATA NAME NOT RECOGNIZED!')
        train_images    = []
        train_labels    = []
        test_images     = []
        test_labels     = []
    
    return train_images, train_labels, test_images, test_labels

def extract_features(train_images, train_labels, test_images, test_labels):
    nClass          = len(set(train_labels))
    nImagePerClass  = [0]*nClass
    train_features  = []
    test_features   = []
    
    for i in range(0,len(train_images)):
        print ('train', i)
        nImagePerClass[train_labels[i]]+=1
        img = resize(train_images[i],(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features_np = np.array(features)
        features_f = features_np.flatten()
        train_features.append(features_f)
    
    for i in range(0,len(test_images)):
        print ('test',i)
        img = resize(test_images[i],(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features_np=np.array(features)
        features_f = features_np.flatten()
        test_features.append(features_f)
    
    train_features  = np.array(train_features)
    test_features   = np.array(test_features)
    train_labels    = np.array(train_labels)
    test_labels     = np.array(test_labels)
    return train_features, train_labels, test_features, test_labels 

def write_features(pDataName, pNetType, pNetFit, train_features, train_labels, test_features, test_labels):
    
    try:        os.mkdir('./features/')
    except:     pass

    writeFile = './features/'+ pDataName + '_' + pNetType + '_' + pNetFit + '_'
    with open(writeFile + 'train_features.data', 'wb') as fh:   pickle.dump(train_features, fh)
    with open(writeFile + 'test_features.data', 'wb') as fh:    pickle.dump(test_features, fh)
    with open(writeFile + 'train_labels.data', 'wb') as fh:     pickle.dump(train_labels, fh)
    with open(writeFile + 'test_labels.data', 'wb') as fh:      pickle.dump(test_labels, fh)


if __name__ == "__main__":

    #inputs
    pDataName               = 'cifar'             
    pNetType                = 'resnet34'            
    pNetFit                 = 'imagenet'    
    
    #data from source
    trainImg, trainLab, testImg, testLab = get_data(pDataName)
    
    #model setup
    net, preprocess_input   = Classifiers.get(pNetType)
    k_model                 = net(input_shape=(224,224,3), weights=pNetFit)
    model                   = Model(inputs = k_model.input, outputs = k_model.get_layer('pool1').output)
    
    #extract features
    trainFeat, trainLab, testFeat, testLab = extract_features(trainImg, trainLab, testImg, testLab)
    
    #write features to file
    write_features(pDataName, pNetType, pNetFit, trainFeat, trainLab, testFeat, testLab)


