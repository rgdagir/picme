import sys
import numpy as np
import imageprocess as imageProcess
import csv
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D, MaxPooling1D, AveragePooling2D
import matplotlib.pyplot as plt
import math
from keras import regularizers
import random
from datetime import datetime 
import textdistance
from keras.utils.vis_utils import plot_model
from skimage.color import rgb2gray
from keras.optimizers import Adam

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
            if (float(row["likeRatio"]) > 1.):
                continue
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
    slashIndex = dataset.find("/")
    slashIndex += 1
    np.save(f"allImgs_{dataset[slashIndex:-4]}.npy", allImgs)
    np.save(f"allResults_{dataset[slashIndex:-4]}.npy", allResults)
    return allImgs, allResults

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

############################################################
# Feature extraction
def extractFeaturesFromDataset(filename):
    net = imageProcess.runFaceDetectDNN()
    print('Start reading features')
    with open(filename) as f:
        featureVectors = []
        results = []
        allImgs = []
        allResults = []
        notProcessed = 0
        totalImgs = 0
        correctShape = 0
        for row in csv.DictReader(f):
            if (float(row["likeRatio"]) > 1.):
                continue
            print(totalImgs)
            totalImgs += 1
            featureVector = []
            somethingFailed = False
            for key in row: #  each row is a dict
                try:
                    if (key == "timestamp"): 
                        hourOfDay = datetime.fromtimestamp(int(row[key])).hour
                        between2and6 = (hourOfDay >= 2 and hourOfDay < 6)
                        between6and10 = (hourOfDay >= 6 and hourOfDay < 10)
                        between10and14 = (hourOfDay >= 10 and hourOfDay < 14)
                        between14and18 = (hourOfDay >= 14 and hourOfDay < 18)
                        between18and22 = (hourOfDay >= 18 and hourOfDay < 22)
                        between22and2 = (hourOfDay >= 22) or (hourOfDay < 2)
                        featureVector.append(int(between2and6))
                        featureVector.append(int(between6and10))
                        # featureVector['between10and14'] = int(between10and14)
                        featureVector.append(int(between14and18)) 
                        featureVector.append(int(between18and22))
                        featureVector.append(int(between22and2))
                
                    
                    elif (key == "caption"):
                        # featureVector["captionLength"] = (len(row[key]))
                        featureVector.append(1 if "food" in row[key].lower() else 0)
                        featureVector.append(1 if "follow" in row[key].lower() else 0)
                        featureVector.append(1 if "ad" in row[key].lower() else 0)
                    
                    # if key == "hashtags":
                    #     hashtags = ast.literal_eval(row[key])
                    #     hashtags = [n.strip() for n in hashtags]
                        # featureVector["numHash"] = 1 if len(hashtags) == 0 else 1./len(hashtags)

                    elif key == "imgUrl":
                        image = imageProcess.Image(row[key], True)
                        imageShape = image.getImageShape()
                        # squaredImage = imageShape[0] == imageShape[1]
                        # isRgb = imageShape[2] == 3;
                        # if (not squaredImage) or (not isRgb):    
                        #     continue
                        # image_rescaled = rescale(image.skimageImage, RESIZE_FACTOR, anti_aliasing=False, multichannel=True)
                        image_rescaled = resize(image.skimageImage, (TARGET_X, TARGET_Y),anti_aliasing=False)
                        featureVector.append(imageProcess.extractSectorsFeature(image, 20, 20))
                        faceInfo = imageProcess.extractFaceInfo(image, net)
                        featureVector.append(imageProcess.extractNumFaces(faceInfo))
                        featureVector.append(imageProcess.extractTotalPercentAreaFaces(faceInfo))
                    elif key == "likeRatio": # we will append the result at the end
                        continue #allResults.append(float(row[key]))
                    elif (key == "likeCount" or key == "commentCount" or key == "timestamp"):
                        featureVector.append(row[key])
                    # this should fail all the time we have a string as the value feature
                    # probably bad style but  python has no better way to check if 
                    # a string contains a float or not
                    else:
                        try:
                            val = float(row[key])
                            featureVector[key] = val
                        except Exception as e:
                            continue
                except Exception as e:
                    somethingFailed = True
                    notProcessed += 1
                    print(e)
                    break
            if (somethingFailed):
                continue
            label = float(row["likeRatio"])
            allResults.append(label)
            allImgs.append(image_rescaled)
            featureVectors.append(featureVector)
        slashIndex = filename.find("/")
        slashIndex += 1
        featureVectors = np.array(featureVectors)
        results = np.array(results)
        np.save(f"allImgs_{filename[slashIndex:-4]}.npy", allImgs)
        np.save(f"allResults_{filename[slashIndex:-4]}.npy", allResults)
        np.save(f"featureVectors_{filename[slashIndex:-4]}.npy", featureVectors)
        return allImgs, featureVectors, allResults



def splitAndPrep(allImgs, allResults):
    # print("shuffling and splitting dataset...")
    # zipped = list(zip(allImgs, allResults))
    # random.shuffle(zipped)
    # allImgs, allResults = list(zip(*zipped))
    (x_train, x_dev, x_test), (y_train, y_dev, y_test) = splitDataset(allImgs, allResults)

    y_train_one_hot = oneHotEncoding(y_train)
    y_test_one_hot = oneHotEncoding(y_test)
    return x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot


def trainModel(modelfilename, x_train, y_train, x_test, y_test):
    #create model
    model = Sequential()

    #add model layers
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(TARGET_X, TARGET_Y,3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))
    # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='softmax'))
                # kernel_regularizer=regularizers.l2(0.01),
                # activity_regularizer=regularizers.l1(0.01)))

    #compile model using accuracy to measure model performance
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='categorical_crossentropy', metrics=['accuracy', 'cosine_proximity'])

    #train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

    # #create model
    # model = Sequential()

    # #add model layers
    # model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(TARGET_X, TARGET_Y,3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # # model.add(Dropout(0.3))
    # model.add(Flatten())
    # model.add(Dense(100, activation='softmax'))
    #             # kernel_regularizer=regularizers.l2(0.01),
    #             # activity_regularizer=regularizers.l1(0.01)))

    # #compile model using accuracy to measure model performance
    # model.compile(optimizer=Adam(lr=1e-4, decay=1e-4/200), loss='categorical_crossentropy', metrics=['accuracy', "cosine_proximity"])

    # #train the model
    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

    print(model.summary())

    plot_model(model, to_file=f"models/{modelfilename}_plot.png", show_shapes=True, show_layer_names=True)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_accuracy.png")

    plt.figure()
    plt.plot(history.history['cosine_proximity'])
    plt.plot(history.history['val_cosine_proximity'])
    plt.title('cosine proximity')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_cosineproximity.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_loss.png")

    model.save(f"models/{modelfilename}.h5")
    return model

def trainModelFeatureVec(modelfilename, x_train, y_train, x_test, y_test):
    #create model
    model = Sequential()
    print(f"xtrain shape {x_train.shape}")
    #add model layers
    model.add(Conv1D(16, kernel_size=3, activation='relu', input_shape=(534,13,1)))
    model.add(MaxPooling1D(pool_size=(2)))
    # model.add(Dropout(0.3))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    # model.add(Dropout(0.3))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    # model.add(Dropout(0.3))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='softmax'))
                # kernel_regularizer=regularizers.l2(0.01),
                # activity_regularizer=regularizers.l1(0.01)))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

    print(model.summary())

    plot_model(model, to_file=f"models/featureVec_{modelfilename}_plot.png", show_shapes=True, show_layer_names=True)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/featureVec_{modelfilename}_accuracy.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/featureVec_{modelfilename}_loss.png")

    model.save(f"models/featureVec_{modelfilename}.h5")
    return model

def test_model(model, x, y):
    indexed_y = list(enumerate(y))
    indexed_y.sort(key=lambda tup: tup[1])
    ideal_ranking = []
    for elem in indexed_y:
        ideal_ranking.append(elem[0])

    predictions = model.predict(x)

    scores = []
    for prediction in predictions:
        scores.append(np.dot(prediction, [i for i in range(1,101)]))
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda tup: tup[1])
    predicted_ranking = []
    for elem in indexed_scores:
        predicted_ranking.append(elem[0])
    if (ideal_ranking == predicted_ranking):
        print("Success! Prediction matched!")
    else:
        mistakes = 0
        for i in range(len(ideal_ranking)):
            if ideal_ranking[i] != predicted_ranking[i]:
                mistakes += 1 
        print(f"We made {mistakes} mistakes. Error ratio: {mistakes/len(ideal_ranking)}")
        print(f"similarity: {textdistance.levenshtein.similarity(ideal_ranking,predicted_ranking)}")
        print(f"distance: {textdistance.levenshtein.distance(ideal_ranking,predicted_ranking)}")
    print(f"ideal_ranking:     {ideal_ranking}")
    print(f"predicted_ranking: {predicted_ranking}\n\n")

def predict(model, x_dev, y_dev):
    print(f"y_dev: {y_dev}")
    actualBestImageIndex = np.argmax(y_dev)

    print(f"theoretical best image index: {actualBestImageIndex}")

    predictions = model.predict(x_dev)

    scores = []
    for prediction in predictions:
        scores.append(np.dot(prediction, [i for i in range(1,101)]))
    scores = np.array(scores)
    predictedImageIndex = np.argmax(scores)
    print(f"scores: {scores}")
    print(f"predictedImageIndex: {predictedImageIndex}")
    plt.figure()
    plt.imshow(x_dev[predictedImageIndex])
    
    #  # WRONG - USE PHIL'S VERSION ABOVE
    # maxBucket = 0 
    # maxProbsForMaxBucket = 0.0
    # predictedImageIndex = 0
    # for i in range(len(predictions)):
    #     prediction = predictions[i]
    #     maxResultIndex = np.argmax(prediction)
    #     maxResult = prediction[maxResultIndex]
    #     if maxBucket <= maxResultIndex:
    #         maxBucket = maxResultIndex
    #         maxProbsForMaxBucket = maxResult
    #         predictedImageIndex = i
    #     elif maxBucket == maxResultIndex:
    #         if maxProbsForMaxBucket < maxResult:
    #             maxProbsForMaxBucket = maxResult
    #             predictedImageIndex = i
    # print(f"predictedImageIndex: {predictedImageIndex}")
    # plt.figure()
    # plt.imshow(x_dev[predictedImageIndex])


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

def concatenatedModelMain():
    if (len(sys.argv) < 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "-d"):
        allImgs, featureVectors, allResults = extractFeaturesFromDataset(sys.argv[2])
        imgX_train, imgX_dev, imgX_test, imgY_train, imgY_dev, imgY_test, imgY_train_one_hot, imgY_test_one_hot = splitAndPrep(allImgs, allResults)
        mdX_train, mdX_dev, mdX_test, mdY_train, mdY_dev, mdY_test, mdY_train_one_hot, mdY_test_one_hot = splitAndPrep(featureVectors, allResults)
        imgX_train = imgX_train.reshape(imgX_train.shape[0], TARGET_X, TARGET_Y, 3)
        imgX_test = imgX_test.reshape(imgX_test.shape[0], TARGET_X, TARGET_Y, 3)
        imageModel = trainModel(extractDatasetNameCSV(sys.argv[2]), x_train, y_train_one_hot, x_test, y_test_one_hot)
        mdModel = trainMdModel()
    elif (sys.argv[1] == "-f"):
        allImgs, allResults = loadFromFile([sys.argv[2], sys.argv[3]])
        # allImgs = grayscaleResize(allImgs)
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = splitAndPrep(allImgs, allResults)        
        try:
            if(sys.argv[4] == "-m"):
                model = load_model(sys.argv[5])
        except:
            model = trainModel(extractDatasetNameNPY(sys.argv[2]), x_train, y_train_one_hot, x_test, y_test_one_hot)    
    else:
        print("Invalid flag, mate!")
        sys.exit(0)
    split_test_set = splitList(x_test, 10)
    split_result_test = splitList(y_test, 10)   
    for i in range(len(split_result_test)):
        test_model(model, split_test_set[i], split_result_test[i])
    # predict(model, x_test, y_test)

if __name__ == "__main__":
    oldmain()


def oldmain()
    if (len(sys.argv) < 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "--full"):
        # featureVectors, results = extractFeaturesFromDataset(sys.argv[2])
        featureVectors, results = loadFromFile([sys.argv[2], sys.argv[3]])
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = splitAndPrep(featureVectors, results)
        model = trainModelFeatureVec("featuresvectormodel", x_train, y_train_one_hot, x_test, y_test_one_hot)
    elif (sys.argv[1] == "-d"):
        allImgs, allResults = downloadImages(sys.argv[2])
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = splitAndPrep(allImgs, allResults)
        x_train = x_train.reshape(x_train.shape[0], TARGET_X, TARGET_Y, 1)
        x_test = x_test.reshape(x_test.shape[0], TARGET_X, TARGET_Y, 3)
        model = trainModel(extractDatasetNameCSV(sys.argv[2]), x_train, y_train_one_hot, x_test, y_test_one_hot)
    elif (sys.argv[1] == "-f"):
        allImgs, allResults = loadFromFile([sys.argv[2], sys.argv[3]])
        # allImgs = grayscaleResize(allImgs)
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = splitAndPrep(allImgs, allResults)        
        try:
            if(sys.argv[4] == "-m"):
                model = load_model(sys.argv[5])
        except:
            model = trainModel(extractDatasetNameNPY(sys.argv[2]), x_train, y_train_one_hot, x_test, y_test_one_hot)    
    else:
        print("Invalid flag, mate!")
        sys.exit(0)
    split_test_set = splitList(x_test, 10)
    split_result_test = splitList(y_test, 10)   
    for i in range(len(split_result_test)):
        test_model(model, split_test_set[i], split_result_test[i])
    # predict(model, x_test, y_test)
