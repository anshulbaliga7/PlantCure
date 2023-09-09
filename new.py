import cv2
import numpy as np
from tensorflow import keras
# from keras import load_model


def model_predict(img_path, model):
    
    #update by ViPS
    img = cv2.imread(img_path)
    print(img.shape)
    new_arr = np.reshape(img,(224,224,3))
    new_arr = np.
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(1,256,256,3)
    print(new_arr.shape)
    
    preds = model.predict(new_arr)
    return preds

model_predict('Test\Apple_Scab\image (902).JPG',keras.models.load_model("my_model.h5"))