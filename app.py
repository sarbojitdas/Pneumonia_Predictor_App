from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

import pickle

app = Flask(__name__)
model = load_model('pneumonia_model.h5')

with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
labels = {v: k for k, v in class_indices.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        image = load_img(filepath, target_size=(150,150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        pred = model.predict(image)[0][0]
        prediction = labels[1 if pred > 0.5 else 0]

        return render_template('index.html', prediction=prediction, image=filepath)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
