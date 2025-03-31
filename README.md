# BeeHealth
This project aims to classify bee health and pollinator habitat quality using computer vision and machine learning with PyTorch. The goal is to support ecological research and awareness by identifying healthy vs. unhealthy bees and pollinator-friendly environments.

## 🚀 Features
- Custom CNN model using PyTorch
- Image preprocessing and data augmentation
- Evaluation metrics (accuracy, confusion matrix)
- Optional Streamlit web demo

## 📂 Project Structure
```
bee-health-classifier/
├── data/               # Datasets (images)
├── notebooks/          # Jupyter notebooks for EDA
├── src/                # Python source code
│   ├── dataset.py      # Custom dataset loader
│   ├── model.py        # CNN architecture
│   ├── train.py        # Training pipeline
│   └── predict.py      # Inference script
├── requirements.txt    # Dependencies
├── README.md           # Project description
└── streamlit_app.py    # Optional web demo
```

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🔍 Example Use
```bash
python src/train.py --epochs 10 --lr 0.001
python src/predict.py --image sample.jpg
```

## 🤝 Contributions
Pull requests welcome! Help support pollinator science 🌼🐝

"""
