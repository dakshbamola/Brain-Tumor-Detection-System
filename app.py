# env\Scripts\Activate.ps1 
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = load_model('D:/Codes/ai/bestmodel.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            uploads_folder = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_folder, filename)
            file.save(file_path)
            
            processed_image = preprocess_image(file_path)
            prediction = model.predict(processed_image)[0][0]
            if round(prediction) == 0:
                result = "No Tumor Detected" 
            else:
                result = "Tumor Detected"
            
            image_url = url_for('uploaded_file', filename=filename)
            
            return render_template('index.html', result=result, image_file=image_url)
        else:
            return render_template('index.html', result="No file uploaded. Please choose a file.")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
