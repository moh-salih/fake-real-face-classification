import torch, os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

import utils.dataset  
from model import MobileNet3Large, MobileNet3Small, ShuffleNet, SqueezeNet
from utils.config import ROOT_DIR

import utils.log


# Set up argument parser
parser = argparse.ArgumentParser(description='Evaluation script for real/fake face classification')
parser.add_argument('--batch_size', type=int, default=224, help='Batch size for evaluation')
parser.add_argument('--dataset', type=str, default='easy', choices=['easy', 'hard'], help='Dataset to evaluate on(easy or hard)')

args = parser.parse_args()

# Hyperparameters
batch_size = args.batch_size
dataset_name= args.dataset # Required for visualization/logging 




transform = transforms.Compose([
  transforms.Resize((224, 224)),  
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if dataset_name == 'hard':
  val_dataset_path = os.path.join(ROOT_DIR, 'data', 'input', 'hard', 'Validation')
elif dataset_name == 'easy':
  val_dataset_path = os.path.join(ROOT_DIR, 'data', 'input', 'easy', 'Validation')


val_dataset = os.path.join(ROOT_DIR, 'data', 'input', 'hard', 'Validation')

dataset = utils.dataset.MyDataset(data_dir=val_dataset_path, transform=transform)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Define Models
models = [
  MobileNet3Large(), MobileNet3Small(), SqueezeNet(), ShuffleNet()
]
paths = [
    os.path.join(ROOT_DIR, 'data', 'output', 'models', 'large', '10.pth'),
    os.path.join(ROOT_DIR, 'data', 'output', 'models', 'small', '10.pth'),
    os.path.join(ROOT_DIR, 'data', 'output', 'models', 'squeeze', '10.pth'),
    os.path.join(ROOT_DIR, 'data', 'output', 'models', 'shuffle', '10.pth'),
]

# The order of paths of weights above is important
for i in range(len(models)):
  models[i].load_weights(paths[i])
  models[i].eval()  # Activate evaluation mode


model_accuracies = {model.model_type: 0 for model in models}
model_f1_scores = {model.model_type: 0 for model in models}
model_precision_scores = {model.model_type: 0 for model in models}

with torch.no_grad():
    # The following metrics are collected per batch
    accuracies = {model.model_type: [] for model in models}
    f1_scores = {model.model_type: [] for model in models}
    precision_scores = {model.model_type: [] for model in models}
    confusion_matrices = {model.model_type: [] for model in models}

    for images, labels in val_loader:
        for model in models:      
            predicted = model.classify(images)
            # Metrics
            accuracies[model.model_type].append(accuracy_score(labels, predicted) * 100)
            f1_scores[model.model_type].append(f1_score(labels, predicted, zero_division=0))
            precision_scores[model.model_type].append(precision_score(labels, predicted, zero_division=0))
            confusion_matrices[model.model_type].append(confusion_matrix(labels, predicted,  labels=[0, 1])) 

    for model in models:
      avg_accuracy = np.mean(accuracies[model.model_type])
      avg_f1_score = np.mean(f1_scores[model.model_type])
      avg_precision = np.mean(precision_scores[model.model_type])
      avg_confusion = np.mean(confusion_matrices[model.model_type], axis=0)
      
      model_accuracies[model.model_type] = avg_accuracy
      model_f1_scores[model.model_type] = avg_f1_score
      model_precision_scores[model.model_type] = avg_precision

      utils.log.visualize_confusion_matrix(avg_confusion, xLabel='Predicted', yLabel='True Label', title=f'Confusion Matrix for {model.model_type} model')
    
      print(f'{model.model_type} avg acc: {avg_accuracy:.1f} avg f1: {avg_f1_score:.1f} avg precision: {avg_precision:.1f} avg confusion matrix: {str(avg_confusion)}\n')
      
utils.log.visualize_metric(model_accuracies, 'Model', 'Accuracy', f'Accuracy for {dataset_name}')
utils.log.visualize_metric(model_precision_scores, 'Model', 'Precision', f'Precision score for {dataset_name}')
utils.log.visualize_metric(model_f1_scores, 'Model', 'F1 score', f'F1 score for {dataset_name}')