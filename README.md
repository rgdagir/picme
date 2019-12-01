# CS 221 Final Project by Phil Pfeffer, Raul Dagir, Victor Yin
## This file shall contain future documentation on the modules here implemented.
### This is a test
MBAs are very quiet

Sites for scikit-image:
- Face detector: https://scikit-image.org/docs/stable/auto_examples/applications/plot_face_detection.html#sphx-glr-auto-examples-applications-plot-face-detection-py

## Instructions:
### Scraping the dataset:
1. Create a csv file with only one column "hashtag", and list all hashtags you want to scrape below it.
2. Call python3 scraper.py csvfilename
3. The scraped dataset will be saved as csvfilenameFULL.csv
### Downloading images/training model/making predictions:
- The raul-imgclassifier.py program takes multiple possible arguments
1. Call python3 raul-imgclassifier.py -d datasetname to download images from that dataset, preprocess everything, train the model and make a prediction given the dev set.
2. Call python3 raul-imgclassifier.py -f imgfilename resultsfilename where both files are ".npy" files. Instead of downloading the images from the dataset, it will load the data from those files, then  train the model and make a prediction given the test set.
3. Call python3 raul-imgclassifier.py -f imgfilename resultsfilename -m modelfilename if you want to run the prediction given a previously trained model, defined by the modelfilename argument.

Have fun!
