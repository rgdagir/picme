import sys
import numpy as np
import imageprocess as imageProcess
import csv
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import math
from sklearn.metrics import log_loss

TARGET_X = 270
TARGET_Y = 270
MODEL_PATH="models/model.h5"

def downloadImages(dataset):
    print('Start reading features')
    with open(dataset) as f:
        allImgs = []
        allResults = []
        notProcessed = 0
        totalImgs = 0
        correctShape = 0
        for row in csv.DictReader(f):
            totalImgs += 1
            try:
                image = imageProcess.Image(row["imgUrl"], True)
                imageShape = image.getImageShape()
                # squaredImage = imageShape[0] == imageShape[1]
                # isRgb = imageShape[2] == 3;
                # if (not squaredImage) or (not isRgb):    
                #     continue
                # image_rescaled = rescale(image.skimageImage, RESIZE_FACTOR, anti_aliasing=False, multichannel=True)
                image_rescaled = resize(image.skimageImage, (TARGET_X, TARGET_Y),anti_aliasing=False)
                correctShape += 1
            except Exception as e:
                notProcessed += 1
                continue
            allImgs.append(image_rescaled)
            allResults.append(float(row["likeRatio"]))
    print(f"not processed: {notProcessed/totalImgs}")
    print(f"correct shape total: {correctShape}")
    print(f"correct shape ratio: {correctShape/totalImgs}")
    np.save("allImgs.npy", allImgs)
    np.save("allResults.npy", allResults)
    return allImgs, allResults

def loadFromFiles():
    allImgs = np.load("allImgs.npy")
    allResults = np.load("allResults.npy")
    return allImgs, allResults

def oneHotEncoding(arr):
    oneHots = []
    indices = []
    for y in arr:
        index = int(y*100)%100
        oneHot = []
        for i in range(100):
            oneHot.append(0 if i != index else 1)
        oneHots.append(oneHot)
    return np.array(oneHots)

def preprocess(allImgs, allResults):
    (x_train, x_dev, x_test), (y_train, y_dev, y_test) = splitDataset(allImgs, allResults)

    y_train_one_hot = oneHotEncoding(y_train)
    y_test_one_hot = oneHotEncoding(y_test)
    return x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot

def trainModel(x_train, y_train, x_test, y_test):
    #create model
    model = Sequential()

    #add model layers
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(TARGET_X, TARGET_Y,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    return model, x_dev, y_dev

def predict(model, x_dev, y_dev):
    actualBestImageIndex = np.argmax(y_dev)
    print(f"theoretical best image index: {actualBestImageIndex}")

    predictions = model.predict(x_dev)
    
    maxBucket = 0
    maxProbsForMaxBucket = 0.0
    predictedImageIndex = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        maxResultIndex = np.argmax(prediction)
        maxResult = prediction[maxResultIndex]
        if maxBucket < maxResultIndex:
            maxBucket = maxResultIndex
            maxProbsForMaxBucket = maxResult
            predictedImageIndex = i
        elif maxBucket == maxResultIndex:
            if maxProbsForMaxBucket < maxResult:
                maxProbsForMaxBucket = maxResult
                predictedImageIndex = i

    print(f"predictedImageIndex: {predictedImageIndex}")
    plt.figure()
    plt.imshow(x_dev[predictedImageIndex])
    plt.figure()
    plt.imshow(x_dev[actualBestImageIndex])
    plt.show()


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
    return (x_train, x_dev, x_test), (y_train, y_dev, y_test)

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "-d"):
        allImgs, allResults = downloadImages('datasets/neuralnet-firstdataset.csv')
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = preprocess(allImgs, allResults)
        model = trainModel(x_train, y_train_one_hot, x_test, y_test_one_hot)
    elif (sys.argv[1] == "-f"):
        allImgs, allResults = loadFromFiles()
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = preprocess(allImgs, allResults)
        model = trainModel(x_train, y_train_one_hot, x_test, y_test_one_hot)
    elif (sys.argv[1] == "-m"):
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = preprocess(allImgs, allResults)
        model = load_model(MODEL_PATH)
    else:
        print("Invalid flag, mate!")
    
    predict(model, x_dev, y_dev)
