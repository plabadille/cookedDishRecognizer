'''
Binary convnet for food image recognizion ("do this picture is a cooked dish or something else?")

Tensorflow-gpu using Keras and lauch in a Jupyter ipython notebook. I use an Anaconda python3.5 environment.
This model run in (....to complete....) with 2 Nvidia 970GTX and 16 Go of ram.
The datasets is composed of 2 classes (~100.000 images each): 
    -cookedDish: Food-101 -- Mining Discriminative Components with Random Forests
    -somethingElse: diverse dataset (Cf. dataset.md) and from Restofolio.com data.

This script have to be executed in a ipython notebook. If not you have to comment every line involving trends and statistic generation as well than the magic link "matplotlib inline" 

This script is based on the blog posts:
"Building powerful image classification models using very little data" from blog.keras.io.
AND
"CatdogNet - Keras Convnet Starter" from https://www.kaggle.com/vshevche/dogs-vs-cats-redux-kernels-edition/catdognet-keras-convnet-starter/run/790364

DIFFERENCES:
- The data have been adapted for the need as well as some adjustment. 
- This script also do cross validation
- Always keep the best weight per epoch depending on the validation loss
- generate more trends (like ROC curve).
- Save the all model (json, weight and model) at each run
- Some refactoring

Note: because of some openCv bugs you may have an exception throwed (error: (-215) ssize.area() > 0 in function cv::resize). If so, make sur:
    -there's only JPEG or PNG images in the training or testing directory.
    -the rows * cols of your image is inferior of 2^31
    -Your color deepness is superior to 8 (for some weird reason..)
'''

#0-IMPORT
import os, cv2, random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

#1-PREPARING DATA
##Dir const
TRAIN_DIR = 'data-set/train/'
TEST_DIR = 'data-set/test/'
SAVE_MODEL_DIR = 'model/'
##Output name const
SAVE_MODEL_NAME = 'cookedDish_Model.hdf5'
SAVE_MODEL_WEIGHT_NAME = 'cookedDish_Weight.hdf5'
SAVE_MODEL_JSON_NAME = 'cookedDish_Metadata.json'
##data const
ROWS = 64
COLS = 64
CHANNELS = 3
##Training const
NB_EPOCH = 50
BATCH_SIZE = 50
##Cross validation const
VALIDATION_PERCENT = 0.45
RANDOM_STATE = 10
##Early stopping const
EARLY_SOPPING_PATIENCE = 5

##whole dataset:
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

path_imageTest = []
def prep_data(images, test):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        if test:
            path_imageTest.append(image_file)
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

train = prep_data(train_images, False)
test = prep_data(test_images, True)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))


#2-GENERATING LABELS (litteral over binary)
labels = []
for i in train_images:
    if 'cookedDish' in i:
        labels.append(1)
    else:
        labels.append(0)

sns.countplot(labels)
sns.plt.title('Test cooked dish or something else')

#3-MODEL BASE
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

def isPlat():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy', 'recall', 'precision'])
    return model

model = isPlat()


#5-TRAIN, VALIDATION AND PREDICT
##Cross validation
print('Splitting data into training and testing')
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=VALIDATION_PERCENT, random_state=RANDOM_STATE)

## Callback for loss logging per epoch
class trendsHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

## Early stopping if the validation loss doesn't decrease anymore
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_SOPPING_PATIENCE, verbose=1, mode='min')
## We just save the best model, not the last one used
filepath = SAVE_MODEL_DIR + SAVE_MODEL_WEIGHT_NAME
model_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, verbose=1, mode='min', save_weights_only=True)

## Generate a ROC Curve
def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print("--- ROC CURVE ---")
    print("False positive rate:")
    print(fpr)
    print("True positive rate:")
    print(tpr)

##Start the learning and validation
def run_isDish():
    history = trendsHistory()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
              validation_data=(X_test, y_test), verbose=2, shuffle=True, callbacks=[history, early_stopping, model_checkpoint])
    #generating result for ROC curve
    y_score = model.predict(X_test)
    generate_results(y_test, y_score[:, 0])
    
    #evaluate (!=validate) the model (with test data)
    predictions = model.predict(test, verbose=0)
    return predictions, history

predictions, history = run_isDish()

#6-DISPLAY MODEL TRENDS
##display loss graph (training and validation) per epoch
loss = history.losses
val_loss = history.val_losses

print("--- Training Losses ---")
print(loss)
print("--- validation Losses ---")
print(val_loss)

##display accuracy graph (training and validation) per epoch
acc = history.acc
val_acc = history.val_acc

print("--- Training Accuracy ---")
print(acc)
print("--- validation Accuracy ---")
print(val_acc)

#7-WE LOAD THE BEST WEIGHT BEFORE PREDICTION
model.load_weights(SAVE_MODEL_DIR + SAVE_MODEL_WEIGHT_NAME)

#8-DISPLAY SOME TEST PREDICTION
count = len(path_imageTest)
for i in range(0,count):
    print("\n" + "Image " + path_imageTest[i] + ":")
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a cooked dish'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is something else'.format(1-predictions[i][0]))

#9-SAVE THE MODEL
model.save(SAVE_MODEL_DIR + SAVE_MODEL_NAME)

model_json = model.to_json()
with open(SAVE_MODEL_DIR + SAVE_MODEL_JSON_NAME, "w") as json_file:
    json_file.write(model_json)
#the best weight have already been saved before
