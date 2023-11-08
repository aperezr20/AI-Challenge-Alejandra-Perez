import argparse
import os
from PIL import Image
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score

class EmotionsDataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.Tensor(self.labels[idx])  # Use one-hot encoded labels
        if self.transform:
            image = self.transform(image)
        return image, label

class EmotionsModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionsModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, num_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.base_model(x)
        x = self.softmax(x)
        return x


def load_dataset(args):
    dataset_path = args.dataset_path
    print("Loading Dataset")
    imgs = []
    labels = []

    class_labels = {
        "anger": [1, 0, 0, 0],
        "happiness": [0, 1, 0, 0],
        "neutrality": [0, 0, 1, 0],
        "sadness": [0, 0, 0, 1]
    }

    for class_name in class_labels.keys():
        class_path = os.path.join(dataset_path, class_name)
        class_images = os.listdir(class_path)
        for image_name in class_images:
            img_path = os.path.join(class_path, image_name)
            if img_path.endswith('.png'):
                imgs.append(img_path)
                labels.append(class_labels[class_name])

    print("Total number of images: {}".format(len(imgs)))
    return imgs, labels

def train(args):
    # Set random seed for reproducibility
    torch.manual_seed(0)

    if args.aug == '':
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    imgs, labels = load_dataset(args)
    X_train, X_val, y_train, y_val = train_test_split(imgs, labels, test_size=0.20, random_state=0)

    train_dataset = EmotionsDataset(X_train, y_train, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = EmotionsDataset(X_val, y_val, transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = EmotionsModelModel(args.num_classes)
    #checkpoint_weights = torch.load('best_model.pt')
    #model.load_state_dict(checkpoint_weights)
    criterion = nn.CrossEntropyLoss()

    opt = args.optimizer
    if opt== 'adam':
        optimizer = optim.Adam(model.parameters(),  lr=args.lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(),  lr=args.lr)
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(),  lr=args.lr) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)

    best_f1 = 0  # Initialize the best AUC score
    best_model_path = None

    start_time = time.time()

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy()) 

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        predicted_classes = np.argmax(all_predictions, axis=1)
        labels_classes = np.argmax(all_labels, axis = 1)

        macro_f1 = f1_score(labels_classes, predicted_classes, average='macro')
        macro_precision = precision_score(labels_classes, predicted_classes, average='macro')
        macro_recall = recall_score(labels_classes, predicted_classes, average='macro')

        print(f"Epoch [{epoch + 1}/{args.num_epochs}] Loss: {loss.item():.4f} Validation F1: {macro_f1:.4f}")

        # Save the model if the current validation F1 is better than the previous best
        if macro_f1 > best_f1:
            print(f'Best F1 macro at epoch {epoch} | F1 macro: {macro_f1} | Precision: {macro_precision} | Recall: {macro_recall}')

            best_f1 = macro_f1

            best_model_path = f"best_model.pt"
            torch.save(model.state_dict(), best_model_path)

    print(f"Training took {time.time() - start_time:.2f} seconds")
    print(f"Best F1 macro: {best_f1:.4f}, Best Model Saved: {best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a multi-label classification model')
    parser.add_argument("--input_type", default="RGB", choices=["RGB", "wavelet"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--dataset_path", default="image_emotions_bw_cropped")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--aug", default='', choices=["horizontal_flip"])

    args = parser.parse_args()
    train(args)
