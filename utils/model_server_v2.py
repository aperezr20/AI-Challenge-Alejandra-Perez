from flask import Flask, request, jsonify
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(0)

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, num_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.base_model(x)
        x = self.softmax(x)
        return x

def classify_input_image(input_image, original_image, emotion_model, resnet, dataset):
    device = 'cpu'
    emotion_model.to(device)
    input_image = input_image.to(device)

    # Perform emotion prediction
    with torch.no_grad():
        input_image = input_image.unsqueeze(0).to(device)
        emotion_logits = emotion_model(input_image)
        _, predicted_emotion_class = torch.max(emotion_logits, 1)

    # Perform face recognition
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
    face, prob = mtcnn(original_image, return_prob=True)
    idx_to_class = dataset[1]

    if face is not None and prob > 0.90:
        emb = resnet(face.unsqueeze(0)).detach()
        embedding_list = dataset[0]  # Use the dataset for face recognition 
        dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
        idx_min = dist_list.index(min(dist_list))
        person_name = idx_to_class[idx_min]
        distance = min(dist_list)
        if distance >= 0.80:
            person_name = 'Unknown'

    else:
        person_name = "Unknown"

    # Return emotion class and person's name
    return predicted_emotion_class.item(), person_name

def load_trained_model(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Instantiation of the model
    emotion_model =  CustomModel(num_classes=4).to(device)
    emotion_model.load_state_dict(torch.load(model_path))
    emotion_model.eval()
    return emotion_model

def preprocess_input_image(input_path):
    # Load the input image as a PIL Image
    input_image = Image.open(input_path).convert('RGB')

    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=True,
    device='cuda:0' if torch.cuda.is_available() else 'cpu')

    # Detect and crop the face using MTCNN
    face = mtcnn(input_image)

    if face is None:
        raise ValueError("No face detected in the input image.")

    to_pil = transforms.ToPILImage()
    face = to_pil(face)
    
    face = face.convert('L')
    face = face.convert('RGB')

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Apply transformations
    preprocessed_face = transform(face)

    return preprocessed_face

app = Flask(__name__)

# Load the model
model_path = 'best_model.pt'
emotion_model = load_trained_model(model_path)

emotions = ['angry', 'happy', 'normal', 'sad']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input image from the request
        image_file = request.files['image']
        original_image  = Image.open(image_file).convert('RGB')
        input_image = preprocess_input_image(image_file)
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        dataset = torch.load('data.pt')

        # Classify the input image
        emotion_class, person_name = classify_input_image(input_image, original_image, emotion_model, resnet,  dataset)

        emotion = emotions[emotion_class]

        # Return the prediction result as JSON
        return jsonify({'emotion': emotion, 'person_name': person_name})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
