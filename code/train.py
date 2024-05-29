import os, argparse
from model import MobileNet3Large, MobileNet3Small, ShuffleNet, SqueezeNet
import numpy as np
from utils.config import ROOT_DIR, check_and_mkdir_if_necessary
import utils.dataset 
import utils.log 

# Set up argument parser
parser = argparse.ArgumentParser(description='Training script for real/fake face classification')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=224, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving the model/metrics')
parser.add_argument('--dataset', type=str, default='easy', choices=['easy', 'hard'], help='Dataset to train on(easy or hard)')

args = parser.parse_args()

# Set Hyperparameters
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
save_interval = args.save_interval
dataset_name= args.dataset


if dataset_name == 'hard':
    train_dataset_path = os.path.join(ROOT_DIR, 'data', 'input', 'hard', 'train')
    # `hard_dataset` is not by default split into train, test, Validation,
    # you might consider uncommenting the below line [SHOULD ONLY BE RUN ONCE]
    # utils.dataset.split_dataset(data_dir=os.path.join(ROOT_DIR, 'data', 'input', 'hard'))
elif dataset_name == 'easy':
    train_dataset_path = os.path.join(ROOT_DIR, 'data', 'input', 'easy', 'train')
        
# Preprocess and Load Datasets
train_dataloader = utils.dataset.preprocess_and_load(dataset_dir=train_dataset_path, batch_size=batch_size)

# Define Models
models = [MobileNet3Large(lr), MobileNet3Small(lr), SqueezeNet(lr), ShuffleNet(lr)]

# Traing
training_losses = {model.model_type: [] for model in models}
training_accuracies = {model.model_type: [] for model in models}

for epoch in range(epochs):
    epoch_losses = {model.model_type: [] for model in models}
    epoch_accuracies = {model.model_type: [] for model in models}

    for model in models:
        model.train() # setting mode per epoch to train has a useful impacts on learning
        
        for images, labels in train_dataloader:
            loss, accuracy = model.train_batch(images, labels)
            
            epoch_losses[model.model_type].append(loss.item())
            epoch_accuracies[model.model_type].append(accuracy)
    
        utils.log.save_model(epoch, model, save_interval=save_interval)
        
    # Compute loss of current epoch, one value for each model  
    for model_name, loss in epoch_losses.items():
            training_losses[model_name].append(np.mean(loss))

    # Compute accuracy of current epoch, also one value for each model  
    for model_name, accuracy in epoch_accuracies.items():
            training_accuracies[model_name].append(np.mean(accuracy))
            
    utils.log.print_models_metrics_per_epoch(epoch, epochs, epoch_accuracies, epoch_losses) 
    
    # The output will be saved to {root}/data/output/train/visualization/
    utils.log.visualize(epoch, training_losses, title=f'Loss over epoch for {dataset_name}'.title(), output_name='loss', save_interval=save_interval)
    utils.log.visualize(epoch, training_accuracies, title=f'Accuracy over epoch for {dataset_name}'.title(), output_name='accuracy', save_interval=save_interval)
    


