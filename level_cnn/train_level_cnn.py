import os
import time
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Hyperparameters
BATCH_SIZE = 32
learning_rate = 0.001
EPOCHS = 5


IMAGE_HEIGHT = 40
IMAGE_WIDTH = 140
CSV_PATH = "../training_data/level/level_labels.csv"
IMAGE_DIR = "../training_data/level"

class LevelImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = int(row['class_id'])

        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    # image transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # load dataset
    dataset = LevelImageDataset(
        csv_file=CSV_PATH,
        image_dir=IMAGE_DIR,
        transform=transform
    )

    # split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # alter ResNet-18 model to small image input
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # smaller convolution on small image
    model.maxpool = nn.Identity()  # remove maxpool layer to not reduce image size
    num_classes = len(dataset.data['class_id'].unique())  # number of unique classes
    model.fc = nn.LazyLinear(num_classes)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(model)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate training loss
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} has Training Loss = {avg_train_loss:.4f}")
    
        # evaluation
        model.eval()
        correct = 0
        total = 0
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                # accumulate validation loss
                total_val_loss +=loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        acc = correct/total*100
        val_accs.append(acc)
        print(f"Validation Accuracy: {acc:.4f}%")
        # save model after each epoch
        torch.save(model.state_dict(), f"level_resnet18_epoch{epoch+1}.pth")

        # adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"Epoch {epoch+1} took {h:02}:{m:02}:{s:.2f}")

    # save model after training
    torch.save(model.state_dict(), "level_cnn_model.pth")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()


if __name__ == "__main__":
    main()