from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__) 



# Model saved with Keras model.save()
MODEL_PATH = 'my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

model.make_predict_function()      
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    #update by ViPS
    img = cv2.imread(img_path)
    print(img.shape)
    new_arr = np.reshape(img,(256,256,3))
    # new_arr = np.
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(1,256,256,3)
    print(new_arr.shape)
    
    preds = model.predict(new_arr)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files.getlist('files[]')
        predictcat={}
        count=1
        results = []
        for file in f:
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads',file.filename )  #secure_filename(f.filename)
            file.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)

            # Process your result for human
            pred_class = preds.argmax()              # Simple argmax
    
            
            CATEGORIES = ['Apple__Apple_scab','Apple__Black_rot','Apple__Cedar_apple_rust','Apple__healthy','Corn__Cercospora_leaf_spot Gray_leaf_spot','Corn__Common_rust','Corn__healthy','Corn__Northern_Leaf_Blight']
            # predictcat.append(CATEGORIES[pred_class])
            predictcat.update({count:CATEGORIES[pred_class]})
            count=count+1
            print("hI",predictcat)

            # html = '<h2>predictcat.values()</h2>'
            # predictcat.values()
            # return CATEGORIES[pred_class]

        # return predictcat
    # return None
    if predictcat[1] == 'Apple__Apple_scab':
        render_template('base.html',data = "Name: Apple Scab\n Info: Apple Scab rarely kills its host, but the damage it does to the fruits and flowers is huge. Apple scab enhances the suscptibility to secondary infection and abiotic stress. The first symptoms of the disease are in the developing fruits of the affected trees. It results to a decrease of yield by almost 70%.\n Prevention: To reduce the scab related yield losses, growers often combine preventive practices, including sanitation, resisitance breeding, with reactive measures such as targetted fungicides, and biocontrol treatments.")
        exit

    return render_template('index.html',data = predictcat)

if __name__ == '__main__':
    app.run(debug=True)

