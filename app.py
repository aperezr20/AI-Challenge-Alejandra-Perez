from flask import Flask, request, render_template, jsonify
import requests
from PIL import Image
from io import BufferedReader
from werkzeug.utils import secure_filename

from PIL import Image, ImageOps


# URL of the model server on the cloud
model_server_url = 'http://34.16.131.11:8080/predict'

# Function to send a POST request to the model server
def predict_with_model_server(image_file):
    try:
        files = {'image': ('image.jpg', image_file)}
        # Send a POST request to the model server
        response = requests.post(model_server_url, files=files)

        if response.status_code == 200:
            result = response.json()
            emotion = result['emotion']
            person_name = result['person_name']
            return emotion, person_name
        else:
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # User image file
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        temp_file_path = filename 

        image_file.save(temp_file_path)

        image = Image.open(temp_file_path)
        image = ImageOps.exif_transpose(image)

        jpg_path =  'static/uploaded_image.jpg'  
        image.save(jpg_path, format='JPEG')

        with open(jpg_path, 'rb') as image_file:
            # Perfom perdiction with model server
            emotion, person_name = predict_with_model_server(image_file)
        
        return render_template('index.html', emotion=emotion, person_name=person_name)

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)

