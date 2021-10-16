# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,BatchNormalization,UpSampling2D
from tensorflow.keras.layers import PReLU,Add,Concatenate,LeakyReLU,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import time
import cv2
import os

#Build ResNet CNN
# current direction
curDir = os.path.dirname(os.path.abspath(__file__))
#resize the image
imgSize = (300,300,3)
#Basic configuration
layerNum = 2
nodeNum = 50
batchSize = 5

# get resBlock
def resBlock(xIn,filterNum,kernelSize=3):
    x = Conv2D(filters=filterNum,kernel_size=kernelSize,padding='same')(xIn)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=filterNum,kernel_size=kernelSize,padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([xIn, x])
    x = LeakyReLU()(x)
    return x

# Build Residual network
# layerNum: the number of residual blocks
# filterNum: the number of residual block convolution kernels
# outNum: output class number
# lr: learning rate
def buildNet(layerNum,filterNum,outNum,lr=1e-4):
    inputLayer = Input(shape=(None,None,3))
    
    #set first layer
    firstLayer = Conv2D(filters=filterNum,kernel_size=3,padding='same')(inputLayer)
    firstLayer = BatchNormalization()(firstLayer)
    firstLayer = LeakyReLU()(firstLayer)  

    #set middle layer
    middle = firstLayer
    for num in range(layerNum):
        middle = resBlock(middle,filterNum)   
    middle = Conv2D(filters=filterNum,kernel_size=3,padding='same')(middle)
    middle = BatchNormalization()(middle)
    middle = LeakyReLU()(middle)
    middle = Add()([firstLayer,middle])
    middle = GlobalAveragePooling2D()(middle)

    #set middle layer
    outputLayer = Dense(outNum,activation='softmax')(middle)    
    
    #construct the model
    model = Model(inputs=inputLayer,outputs=outputLayer)    
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#construct the model
net = buildNet(layerNum,nodeNum,2)
net.summary()

# 0:off-topic essays; 1:on-topic essays
def collect():
    #process the image data
    data = {"imgPath":[],"filename":[]}
    for root, dirs, files in os.walk(curDir+"/图片"):
        for file in files:
            path = os.path.join(root, file).replace("\\","/")
            data["imgPath"].append(path)
            data["filename"].append(file[:-4].replace("temp",""))
    data = pd.DataFrame(data)
    data["txtPath"] = None
    data["label"] = None
    data["flag"] = None
    for root, dirs, files in os.walk(curDir+"/data"):
        for file in files:
            path = os.path.join(root, file).replace("\\","/")
            index = data[data["filename"]==file].index
            if "/0/" in path:
                label = 0
            if "/1/" in path:
                label = 1
            if np.random.uniform(0,1)>0.8:
                flag = "test"
            else:
                flag = "train"
            data.loc[index,"label"] = label
            data.loc[index,"txtPath"] = path
            data.loc[index,"flag"] = flag
    return data

data = collect()
print("current data distribution of 0-1")
print(data["label"].value_counts())

def readImg(path):
    '''return a matrix'''
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
    img = (img - 127.5)/127.5
    return img

def cutImg(img):
    '''resize the image'''
    zeros = np.zeros(imgSize)
    img = img[:imgSize[0],:imgSize[1]]
    zeros[:img.shape[0],:img.shape[1]] = img
    return zeros

def readCutImg(path):
    img = readImg(path)
    img = cutImg(img)
    return img

def getBatch():
    '''get training data'''
    xBatch = []
    yBatch = []
    trainData = data[data["flag"]=="train"]
    for label in range(2):
        #get subdata
        subData = trainData[trainData["label"]==label].copy()
        #get batches o fdata
        indexs = np.random.choice(range(subData.shape[0]),batchSize,replace=True)
        subData = subData.iloc[indexs]
        subData["x"] = subData["imgPath"].apply(readCutImg)
        x = subData["x"].values.tolist()
        x = np.stack(x,axis=0)
        y = subData["label"].values.reshape(-1).tolist()
        y = to_categorical(y,2)
        xBatch.append(x)
        yBatch.append(y)
    xBatch = np.concatenate(xBatch,axis=0)
    yBatch = np.concatenate(yBatch,axis=0)
    return xBatch,yBatch

#allocate GPU resources
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

#Network evaluation
def evalModel(model,flag="test"):
    if flag=="val":
        data_ = data[data["flag"]=="val"]
    elif flag=="test":
        data_ = data[data["flag"]=="test"]
    accList = []
    batchSize2 = 5*batchSize
    for i in range(0,data_.shape[0],batchSize2):
        if i>=data_.shape[0]:
            break
        subData = data_.iloc[i:i+batchSize2].copy()
        subData["x"] = subData["imgPath"].apply(readCutImg)
        x = subData["x"].values.tolist()
        x = np.stack(x,axis=0)
        y = subData["label"].values.reshape(-1).tolist()
        y = to_categorical(y,2)
        yPred = model.predict(x,batch_size=batchSize2)
        yTrue = y.argmax(axis=1)
        yPred = yPred.argmax(axis=1)
        acc = accuracy_score(yTrue,yPred)
        accList.append(acc)
    acc = np.mean(accList)
    return acc

epoch = 0
history = {"epoch":[],"trainLoss":[],"trainAcc":[],"testAcc":[]}
trainLossList = []
trainAccList = []
bestTestAcc = 0

while True:
    xBatch,yBatch = getBatch()
    trainLoss,trainAcc = net.train_on_batch(xBatch,yBatch)
    trainLossList.append(trainLoss)
    trainAccList.append(trainAcc)
    del xBatch,yBatch
    if epoch%50==0:
        testAcc = evalModel(net,"test")
        #save the best model
        if testAcc>bestTestAcc:
            bestTestAcc = testAcc
            net.save_weights(curDir+"/best.h5")
        #record the training history
        trainLossMean = np.mean(trainLossList[-20:])
        trainAccMean = np.mean(trainAccList[-20:])
        history["epoch"].append(epoch)
        history["trainLoss"].append(trainLossMean)
        history["trainAcc"].append(trainAccMean)
        history["testAcc"].append(testAcc)
        print("epoch:%d trainLoss:%.2f trainAcc:%.2f testAcc:%.2f"%(epoch,trainLossMean,trainAccMean,testAcc))
        historyDF = pd.DataFrame(history)
        historyDF.to_csv(curDir+"/training_history.csv",index=None)
    epoch += 1


