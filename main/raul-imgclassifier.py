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

def downloadImages(dataset):
    print('Start reading features')
    with open(dataset) as f:
        allImgs = []
        allResults = []
        notProcessed = 0
        totalImgs = 0
        correctShape = 0
        for row in csv.DictReader(f):
            print(totalImgs)
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
            except Exception as e:
                notProcessed += 1
                print(e)
                continue
            allImgs.append(image_rescaled)
            allResults.append(float(row["likeRatio"]))
    print("not processed: " + str(notProcessed/totalImgs))
    np.save(f"allImgs_{dataset[9:-4]}.npy", allImgs)
    np.save(f"allResults_{dataset[9:-4]}.npy", allResults)
    return allImgs, allResults

def loadFromFile(filename):
    allImgs = np.load(filename)
    allResults = np.load(filename)
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
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='softmax'))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
    model.save(MODEL_PATH)
    return model

def predict(model, x_dev, y_dev):
    print(f"y_dev: {y_dev}")
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
        if maxBucket <= maxResultIndex:
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
    if (len(sys.argv) < 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "-d"):
        allImgs, allResults = downloadImages(sys.argv[2])
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = preprocess(allImgs, allResults)
        model = trainModel(x_train, y_train_one_hot, x_test, y_test_one_hot)
    elif (sys.argv[1] == "-f"):
        allImgs, allResults = loadFromFile(sys.argv[2])
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = preprocess(allImgs, allResults)
        
        if (sys.argv[3] == "-m"):
            allImgs, allResults = loadFromFile(sys.argv[4])
            x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = preprocess(allImgs, allResults)
            model = load_model(sys.argv3)
        else:
            model = trainModel(x_train, y_train_one_hot, x_test, y_test_one_hot)    
    else:
        print("Invalid flag, mate!")
        sys.exit(0)
    
    predict(model, x_test, y_test)
