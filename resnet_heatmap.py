import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from keras.optimizers import SGD
from keras.models import load_model
from utils import getFeatures, loadData, getAccuracy
from keras.models import Sequential
from keras.layers import Dense, add
from scale_layer import Scale
from resnet_152 import resnet152_model

class ImageInput(object):
    def __init__(self, file_input, labels_input,
                model_input, overlapx, overlapy):
        self.file_input = file_input
        self.labels_input= labels_input
        self.model_input = model_input
        self.overlapx = overlapx
        self.overlapy = overlapy

def sliding_window(imageinput):
    posx = 0
    posy = 0
    cont = 0

    classes = next(os.walk('1_Dataset/train'))[1]
    train, train_labels = loadData('1_Dataset/train', 224, 224, classes)
    test, test_labels = loadData('1_Dataset/test', 224, 224, classes)
    model_features = resnet152_model(224, 224, 3, len(classes))
    features = getFeatures(model_features, train, 8)
    test_features = getFeatures(model_features, test, 8)
    prediction_model = Sequential()
    
    prediction_model.add(Dense(256, input_shape = features.shape[1:],  activation='relu'))
    prediction_model.add(Dense(len(classes), activation='softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    prediction_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    prediction_model.fit(features, train_labels, epochs = 10, batch_size = 8)

    predictions = prediction_model.predict(test_features, batch_size = 8, verbose=1)
    
    acc = getAccuracy(test_labels, predictions)
    print(acc)

    if os.path.isfile(imageinput.file_input):
        image = Image.open(imageinput.file_input)
        size = image.size
        width = size[0]
        height = size[1]
        csv_path = os.path.splitext(imageinput.file_input)[0] + '.csv'
        csv_file = open(csv_path, 'w')


        for divisor in range(40,41,1):
            posx = 0
            posy = 0
            new_width = width / divisor
            new_height = height / divisor
            overlap_x = new_width * imageinput.overlapx
            overlap_y = new_height * imageinput.overlapy

            new_path = 'solution/divisor_' + str(divisor)
            if not os.path.isdir(new_path):
                os.makedirs(new_path)

            while new_height <= height:
                posx = 0
                new_width = width / divisor
                while new_width <= width:
                    cont += 1
                    box = (posx, posy, new_width, new_height)
                    region = image.crop(box)
                    new_name = new_path + '/' + str(cont) + 'image_' +str(posx) + '_' + str(posy) + '.jpg'
                    region.save(new_name)

                    img = np.array([cv2.resize((cv2.imread(new_name)).astype(np.float32),
                        (224, 224))])

                    img_features = getFeatures(model_features, img, 8)

                    predictions = prediction_model.predict(img_features, batch_size=8,verbose=1)

                    csv_file.write('%i,Z,%f,%i,%i,%i,%i,%i,%i,%i,%i,%i\n' % (cont,predictions[0][0],divisor,posx,posy,posx,new_width,new_width,new_height,new_height,posy))
                    csv_file.write('%i,S,%f,%i,%i,%i,%i,%i,%i,%i,%i,%i\n' % (cont,predictions[0][1],divisor,posx,posy,posx,new_width,new_width,new_height,new_height,posy))
                    posx += overlap_x
                    new_width += overlap_x
                posy += overlap_y
                new_height += overlap_y
        image.close()
    else:
        print("image not found")
        sys.exit(1)

def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-i", "--image", required=True, help="path to image")
    argp.add_argument("-m", "--model", required=True, help="path to trained model")
    argp.add_argument("-l", "--label", required=True, help="path to labels text file")
    args = vars(argp.parse_args())

    image = ImageInput(args["image"], args["label"], args["model"], 1, 1);
    sliding_window(image)




    

if __name__ == "__main__":
    main()
