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
from services import preprocess_service

app = Flask(__name__)

model = CNN()
model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
model.eval()



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()

    input_tensor = preprocess_service.process_image(data.get('image'))

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
