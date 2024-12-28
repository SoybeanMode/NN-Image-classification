import torch
import torch.nn as nn
import PIL
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 512)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 37 * 37)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)), 
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  
    return image


def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted = (outputs > 0.5).float()
    return predicted.item()


model_path = 'C:\\Users\\sawye\\OneDrive\\Desktop\\jupiter\\mymodel.pth'
image_path = 'C:\\Users\\sawye\\OneDrive\\Desktop\\jupiter\\images\\summer2.jpg'

model = load_model(model_path)
image_tensor = preprocess_image(image_path)
print(image_tensor)
prediction = predict(model, image_tensor)
print(f"Predicted class: {prediction}")
