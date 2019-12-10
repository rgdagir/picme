import numpy as np
import math
from skimage.color import rgb2gray
from datetime import datetime

def ep_to_day(ep):
    return datetime.fromtimestamp(ep/1000).strftime("%A")

def splitAndPrep(allImgs, allResults):
    # print("shuffling and splitting dataset...")
    # zipped = list(zip(allImgs, allResults))
    # random.shuffle(zipped)
    # allImgs, allResults = list(zip(*zipped))
    (x_train, x_dev, x_test), (y_train, y_dev, y_test) = splitDataset(allImgs, allResults)

    y_train_one_hot = oneHotEncoding(y_train)
    y_test_one_hot = oneHotEncoding(y_test)
    return x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot

def loadFromFile(files):
    out = []
    for f in files:
        try:
            out.append(np.load(f))
        except Exception as e:
            print(e)
    return out
    
def oneHotEncoding(arr):
    oneHots = []
    for y in arr:
        index = int(y*100)%100
        oneHot = []
        for i in range(100):
            oneHot.append(0 if i != index else 1)
        oneHots.append(oneHot)
    return np.array(oneHots)

def splitDataset(allImgs, allResults):
    limitTrain = 6*int(len(allImgs)/10)
    x_train = np.array(allImgs[:limitTrain])
    y_train = np.array(allResults[:limitTrain])
    prevx_test = np.array(allImgs[limitTrain:])
    prevy_test = np.array(allResults[limitTrain:])
    limitTest = int(len(prevx_test)/2)
    x_dev = np.array(prevx_test[limitTest:])
    y_dev = np.array(prevy_test[:limitTest])
    x_test = np.array(prevx_test[:limitTest])
    y_test = np.array(prevy_test[:limitTest])
    # x_train = x_train.reshape(len(x_train),TARGET_X, TARGET_Y,1)
    # x_test = x_test.reshape(len(x_test),TARGET_X, TARGET_Y,1)
    # x_dev = x_test.reshape(len(x_dev),TARGET_X, TARGET_Y,1)
    return (x_train, x_dev, x_test), (y_train, y_dev, y_test)

def splitList(lst, sz):
    lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    return lol(lst, sz)

def extractDatasetNameNPY(datafilename):
    extracted = datafilename[:-4] # removing .csv
    indexUnderscore = datafilename.find("_")
    indexUnderscore += 1
    extracted = extracted[indexUnderscore:]
    return extracted

def extractDatasetNameCSV(datafilename):
    extracted = datafilename[:-4] # removing .csv
    slashIndex = datafilename.find("/")
    slashIndex += 1
    extracted = extracted[slashIndex:]
    return extracted

def grayscaleResize(allImgs):
    print("greyscaling and resizing...")
    graySmallImgs = []
    for img in allImgs:
        grayscale = rgb2gray(img)
        image_rescaled = resize(grayscale, (TARGET_X, TARGET_Y),anti_aliasing=False)
        graySmallImgs.append(image_rescaled)
    rgb_batch = np.repeat(graySmallImgs, 3)
    rgb_batch = rgb_batch.reshape(len(graySmallImgs),TARGET_X, TARGET_Y,3)
    return np.array(rgb_batch)