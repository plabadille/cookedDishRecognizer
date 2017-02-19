'''
This demo will only give you textuel output. If you want to actually see your image, use the demo.py script in a Jupyter notebook.
This model handle jpg and png image only.
Just run it with your test file in /predictDemo/*

Note: because of some openCv bugs you may have an exception throwed (error: (-215) ssize.area() > 0 in function cv::resize). If so, make sur:
    -there's only JPEG or PNG images in the predictDemo folder.
    -the rows * cols of your image is inferior of 2^31
    -Your color deepness is superior to 8 (for some weird reason..)
'''

#1-IMPORT
import os, cv2, random
import numpy as np

from keras.models import Sequential, model_from_json

##Dir const
PREDICT_DIR = 'data-set/predictDemo/'
SAVE_MODEL_DIR = 'model/'
##Input name const
SAVE_MODEL_WEIGHT_NAME = 'cookedDish_Weight.hdf5'
SAVE_MODEL_JSON_NAME = 'cookedDish_Metadata.json'
##data const
ROWS = 64
COLS = 64
CHANNELS = 3

#2-RELOAD BEST MODEL
## We load the model from the json save
print("\n-------------------------------")
print("----Reloading model from save----")
print("-------------------------------\n")
json_file = open(SAVE_MODEL_DIR + SAVE_MODEL_JSON_NAME, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
## We load the best weight from hdf5 save
model.load_weights(SAVE_MODEL_DIR + SAVE_MODEL_WEIGHT_NAME)
print("\n----------------------")
print("-------Model load-----")
print("----------------------\n")

#3-DO PREDICTION
path_image = []
def read_image(file_path):
    path_image.append(file_path)
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data, count

test_images =  [PREDICT_DIR+i for i in os.listdir(PREDICT_DIR)]
test, count = prep_data(test_images)
predictions = model.predict(test, verbose=0)

print("\n-------------------------")
print("-------PREDICTION:-------")
print("-------------------------")

for i in range(0,count):
    print("\n" + "Image " + path_image[i] + ":")
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a cooked dish'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is something else'.format(1-predictions[i][0]))
