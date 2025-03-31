import torch
from torchvision import transforms
from PIL import Image
from model import BeeClassifierCNN
import sys

image_path = sys.argv[1]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

model = BeeClassifierCNN()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()
    print(f"Predicted Class: {prediction}")