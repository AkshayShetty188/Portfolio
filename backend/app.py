from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F
from torchvision import models
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow frontend JS to access API

# ✅ Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11
class_names = ['boron', 'calcium', 'healthy', 'iron', 'magnesium', 'manganese',
               'nitrogen', 'phosphorus', 'potassium', 'sulphur', 'zinc']

# ✅ Load All Models
def load_model(model_name, path):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unsupported model")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_dir = "D:/Downloads/clark-master/akshaydshetty/model/"
model1 = load_model('efficientnet_b0', os.path.join(model_dir, 'best_model_efficientnet_b0.pth'))
model2 = load_model('resnet50', os.path.join(model_dir, 'best_model_resnet50.pth'))
model3 = load_model('densenet121', os.path.join(model_dir, 'best_model_densenet121.pth'))

# ✅ Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs1 = F.softmax(model1(input_tensor), dim=1)
        probs2 = F.softmax(model2(input_tensor), dim=1)
        probs3 = F.softmax(model3(input_tensor), dim=1)

        avg_probs = (probs1 + probs2 + probs3) / 3
        confidence, predicted = torch.max(avg_probs, 1)

    return jsonify({
        'prediction': class_names[predicted.item()],
        'confidence': round(confidence.item() * 100, 2)
    })

@app.route("/", methods=["GET"])
def home():
    return "Ensemble Plant Nutrient Classifier API Running"

if __name__ == "__main__":
    app.run(debug=True)
