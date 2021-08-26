import os

import numpy as np
import pickle
import sklearn

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

# Initialize prediction models and labels
# feat_extract = load_model('static/feat_extract.h5')
# rf_model = pickle.load(open('static/rf_model.pkl', 'rb'))
# svm_model = pickle.load(open('static/svm_model.pkl', 'rb'))
cnn_model = load_model('static/cnn_model.h5')
class_labels = ['Dried Basil', 'Dried Oregano', 'Dried Parsley', 'Dried Thyme', 'Not a Spice']

@app.route('/')
def index():
    return render_template('index.html', class_pred='Prediction Appears Here')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return render_template('index.html', class_pred='ERROR: Use .jpg or .png')
        else:
            # Prepare raw input image for prediction by resizing and converting to array
            img = Image.open(uploaded_file)
            if file_ext == '.png':
                img = img.convert('RGB')
            resized_img = img.resize((200,200))
            array_img = np.array(resized_img)
            array_img = np.expand_dims(array_img, axis=0)

            # Extract features, predict with the three models, and final ensemble prediction
            # extracted_feat = feat_extract(array_img)
            # rf_pred = rf_model.predict_proba(extracted_feat)
            # svm_pred = svm_model.predict_proba(extracted_feat)
            cnn_pred = cnn_model.predict(array_img)
            # pred = cnn_pred*2 + rf_pred + svm_pred
            pred = cnn_pred

            # Convert numerical prediction to string
            class_pred = class_labels[np.argmax(pred)]
                
            return render_template('index.html', class_pred=class_pred)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)