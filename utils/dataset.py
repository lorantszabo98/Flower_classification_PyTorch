from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random


class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = ImageFolder(root=self.root_dir, transform=self.transform)
        self.classes = self.image_folder.classes

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img_path, label = self.image_folder.imgs[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(image_size, show_image=False):
    # Set your transform for both train and validation datasets
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])

    # Similarly, create DataLoader for the test set
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load the entire dataset
    full_dataset = FlowerDataset(root_dir='./data/flowerdataset/train/', transform=train_transform)

    if show_image:
        idx = random.randint(1, len(full_dataset) - 1)
        image_t, label = full_dataset[idx]
        plt.imshow(image_t.permute(1, 2, 0))
        plt.title(label)
        plt.show()

    # Define the size of the validation set (e.g., 20%)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoader for training set
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Create DataLoader for validation set
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    test_dataset = FlowerDataset(root_dir='./data/flowerdataset/test/', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
