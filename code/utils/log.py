import torch, os
import numpy as np
import matplotlib.pyplot as plt
from utils.config import ROOT_DIR,check_and_mkdir_if_necessary

def save_model(epoch, model, save_interval=10):
    if epoch % save_interval != 0: return
    
    path = os.path.join(ROOT_DIR, 'data', 'output', 'models', model.model_type)
    
    check_and_mkdir_if_necessary(path)
    
    torch.save(model.static_dict(), f'{path}/{epoch + 1}.pth')

def visualize(epoch, values, title, output_name='loss', colors=['red', 'purple', 'orange', 'blue'], save_interval=10):
    if epoch % save_interval != 0: return
    i = 0
    plt.clf()
    for model_name, loss in values.items():
        plt.plot(range(epoch + 1), loss, label=model_name, color=colors[i])
        i += 1

    plt.ylim(bottom=0)

    plt.xlabel('Epoch')
    plt.ylabel(output_name.title())

    plt.title(title)
    plt.legend()

    plt.grid(True)

    path = os.path.join(ROOT_DIR, 'data', 'output', 'train', 'visualization')
    check_and_mkdir_if_necessary(path)  

    plt.savefig(f'{path}/{output_name}_{epoch + 1}.png')

def print_metrics_per_batch(epoch, epochs, accuracies):
    s = f'Epoch {epoch + 1}/{epochs} '
    
def print_model_metrics_per_epoch(epoch, epochs, losses, accuracies):
    average_loss = np.nanmean(losses)
    average_accuracy = np.nanmean(accuracies)

    s = f'Epoch {epoch + 1}/{epochs} '
    s += f'loss: {average_loss:0.1f} accuracy: {average_accuracy:0.1f}'
    
    print(s)

def print_models_metrics_per_epoch(epoch, epochs, accuracies, losses):
    s = f'Epoch {epoch + 1}/{epochs} '
    for model_name, model_accuracy in accuracies.items():
        average_accuracy = np.nanmean(model_accuracy)
        average_loss = np.nanmean(losses[model_name])
        s+=  f'{model_name}: acc. {average_accuracy:0.1f}, loss {average_loss:0.2f}  '
    
    print(s)

def visualize_metric(items, xLabel, yLabel, title):
    colors = np.random.rand(len(items), 3) 
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.bar(items.keys(), items.values(), color=colors)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    
    path = os.path.join(ROOT_DIR, 'data', 'output','eval' , 'visualization')
    check_and_mkdir_if_necessary(path)
    plt.savefig(f'{path}/{title.replace(" ", "_")}.png')

    # plt.show()

def visualize_confusion_matrix(cm, xLabel, yLabel, title='Confusion Matrix'):
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title.title())

    plt.colorbar()
    tick_marks = np.arange(2)

    xClasses = []
    yClasses = []

    plt.xticks(tick_marks, xClasses)
    plt.yticks(tick_marks, yClasses)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    path = os.path.join(ROOT_DIR, 'data', 'output', 'eval', 'visualization')
    check_and_mkdir_if_necessary(path)

    plt.savefig(f'{path}/{title.replace(" ", "_")}.png')

    # plt.show()