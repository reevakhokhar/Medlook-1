import os
from flask import Flask, flash, request, redirect, url_for, render_template

from werkzeug.utils import secure_filename
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageOps

# CONSTANTS
UPLOAD_FOLDER = 'static/uploads/'
# Allowed image extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
# Prediction threshold
PREDICTION_THRESHOLD = .9
# Comparison item
COMPARISON_ITEM = 'medicine'
testing = "hello"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# A function to check for allowed files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask routing HTML pages 
@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename): 
      filename = secure_filename(file.filename)
      filepath = os.path.join(UPLOAD_FOLDER, filename)
      file.save(filepath)
      prediction = process_file(filepath)
      return render_template('index.html', filename=filename, prediction=prediction)
    else:
      return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def process_file(filepath):
    interpreter = tflite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create the array of the right shape to feed into the keras model
    input_shape = input_details[0]['shape']
    input_data = np.ndarray(shape=input_shape, dtype=np.float32)
    
    # Resize image to be at least 224x224 and then cropping from center, turn image to numpy array, normalize the image
    image = Image.open(filepath)
    size= input_shape[1:3]
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0)-1
    input_data[0] = normalized_image_array
   
    # Load the image into the array, run the inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data

    # Prediction text
    prediction_text = truncate(prediction.item(0)*100, 5)
    if prediction.item(0) > PREDICTION_THRESHOLD:
      return "Please enter medicines that is in our database! go to bit.ly/http://bit.ly/medlooksample for examples."
    elif prediction.item(1) > PREDICTION_THRESHOLD:
      return "You're holding Actifed, which is a cough medicine. Drink 3x 5ml in a day for adults and 3x 2.5ml for age 6-12."
    elif prediction.item(2) > PREDICTION_THRESHOLD:
      return "You're holding CDR a vitamin C tablet. Mix it with water and drink it once a day"
    elif prediction.item(3) > PREDICTION_THRESHOLD:
      return "You're holding Tianshi Zinc a Zinc supplement. Drink 3x1 capsule every single day."
    

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])
    
app.run(host='0.0.0.0', port=8080)