import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import splitfolders


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in ["real", "fake"]:
            label_path = os.path.join(data_dir, label)
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                self.image_paths.append(image_path)
                self.labels.append(1 if label == 'real' else 0)

       
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    
def split_dataset(data_dir, ratio=(0.8, 0.1, 0.1)):
    splitfolders.ratio(data_dir, output=data_dir, ratio=ratio, seed=1997, group_prefix=None)
    os.rename(os.path.join(data_dir, 'val'), os.path.join(data_dir, 'Validation'))

def preprocess_and_load(dataset_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    dataset = MyDataset(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader