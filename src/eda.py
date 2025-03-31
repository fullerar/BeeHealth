import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set up dataset path and transformation
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'train')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Class names
class_names = dataset.classes
print(f"Classes found: {class_names}")

# Count images per class
from collections import Counter
import pandas as pd
labels = [label for _, label in dataset]
label_counts = Counter(labels)
class_counts = {class_names[i]: label_counts[i] for i in label_counts}
print("\nImage count per class:")
print(pd.DataFrame(class_counts.items(), columns=['Class', 'Count']))

# Visualize a batch of sample images
images, labels = next(iter(loader))
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i].permute(1, 2, 0))
    ax.set_title(class_names[labels[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()