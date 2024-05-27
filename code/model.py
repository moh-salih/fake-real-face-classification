import torch 
from torch import nn
from torchvision import models

class Classifier:
    def __init__(self, model_type, learning_rate):
        self.model_type = model_type
        

        if model_type == 'large':
            pretrained_model = models.mobilenet_v3_large(pretrained=True, weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        elif model_type == 'small':
            pretrained_model = models.mobilenet_v3_small(pretrained=True, weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        elif model_type == 'squeeze':
            pretrained_model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        elif model_type == 'shuffle':
            pretrained_model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)    
        else:
            raise ValueError("Invalid model_type. It should be either 'small' or 'large'.")

        pretrained_model.float()
        self.model = pretrained_model

        # Freeze all layers except the final classifier layer
        for param in self.model.parameters():
            param.requires_grad = False

        num_classes = 2
        
        if model_type in ["small", "large"]:
            # Make the parameters of the final classifier layer trainable
            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True


            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

        elif model_type == 'squeeze':
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1) 
            self.model.num_classes = num_classes  # Update the number of output classes
        
        elif model_type == 'shuffle':
            self.model.fc = nn.Linear(2048, num_classes)  # Modify the fully connected layer
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.BCELoss()
    

    def train_batch(self, images, labels):
        output = self.model(images)
        outputs_probs = torch.sigmoid(output)

        # Assuming labels are integers (0 for fake, 1 for real)
        # Convert labels to one-hot encoding
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).float()

        loss = self.criterion(outputs_probs, labels_onehot)
        
        
        self.optimizer.zero_grad()
        loss.backward() # Backpropogation
        
        self.optimizer.step() # Update model weights
        

        # Calculate accuracy per batch
        _, predicted = torch.max(output.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        
        accuracy = (correct / total) * 100
        return loss, accuracy
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
    
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def static_dict(self):
        return self.model.state_dict()
    
    def classify(self, images):
        outputs = self.model(images)
        _, predicted = torch.max(outputs, 1)
        return predicted
         
class MobileNet3Large(Classifier):
    def __init__(self, lr=0.001):
        super().__init__(model_type='large', learning_rate=lr)
        

class MobileNet3Small(Classifier):
    def __init__(self, lr=0.001):
        super().__init__(model_type='small', learning_rate=lr)
        
            

class SqueezeNet(Classifier):
    def __init__(self, lr=0.001):
        super().__init__(model_type='squeeze', learning_rate=lr)
        
class ShuffleNet(Classifier):
    def __init__(self, lr=0.001):
        super().__init__(model_type='shuffle', learning_rate=lr)
        
        


