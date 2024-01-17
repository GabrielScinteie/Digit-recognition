from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch.nn.functional as F
import torch
from torchvision import transforms
from model import CNN
import io
import base64
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

model = CNN()
model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
model.eval()

upload_folder = 'uploaded_images'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

def process_image(data_url):
    np.set_printoptions(threshold=np.inf)
    header, image_data = data_url.split(',', 1)
    image_binary = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_binary))

    grayscale_pixel_matrix = np.array(image.convert('L'))
    inverted_pixel_matrix = 255 - grayscale_pixel_matrix
    inverted_image = Image.fromarray(inverted_pixel_matrix.astype('uint8'))

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    processed_image = transform(inverted_image)
    processed_pixel_matrix = np.array(processed_image)
    input_tensor = processed_image.unsqueeze(0)

    image_filename = os.path.join(upload_folder, 'uploaded_image.png')
    inverted_image.save(image_filename)

    return input_tensor

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()

    input_tensor = process_image(data.get('image'))

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    return jsonify({
        'predicted_label': predicted.item(),
        'probability_distribution': probabilities.numpy().tolist(),
    })


if __name__ == '__main__':
    app.run(debug=True)
