import os
import sys
import time
import re
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from rich.progress import track

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from plot_metrics import draw_plot, save_for_plot

# Hyperparameters
BATCH_SIZE = 32
learning_rate = 0.001

DIMENSIONS = {
    # width, height of image respectively
    "level": [140, 40],
    "mainstat": [412, 40],
    "substat": [412, 40]
}

class ImageDataset(Dataset):
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
    if len (sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python train_cnns.py <level|mainstat|substat> [epochs=10] [model_checkpoint_epochX.pth]")
        exit()
    
    cnn_name = sys.argv[1]
    if cnn_name not in DIMENSIONS:
        print("Invalid cnn_name. Provide 'level', 'mainstat', or 'substat'")
        exit()
    
    epochs = 10  # default epochs
    checkpoint_path_position = 3
    # get epochs from command line argument
    if sys.argv[2].isdigit():
        epochs = int(sys.argv[2])
    else:
        checkpoint_path_position = 2
        print(f"Using default epochs: {epochs}")
    
    
    checkpoint_path = f"{cnn_name}_cnn/{sys.argv[checkpoint_path_position]}" if len(sys.argv) == checkpoint_path_position+1 else None
    print(f"Training CNN for {cnn_name} with checkpoint: {checkpoint_path if checkpoint_path else 'None'}")
    
    # image transformations
    transform = transforms.Compose([
        transforms.Resize((DIMENSIONS[cnn_name][1], DIMENSIONS[cnn_name][0])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # load dataset
    dataset = ImageDataset(
        csv_file="training_data/"+cnn_name+"/"+cnn_name+"_labels.csv",
        image_dir="training_data/"+cnn_name,
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

    # load checkpoint if provided
    print(checkpoint_path)
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Checkpoint loaded successfully.")
        match = re.search(r'_epoch(\d+)\.pth', checkpoint_path)
        if match:
            start_epoch = int(match.group(1))
            print(f"Resuming training from epoch {start_epoch + 1}, for {epochs} epochs.")
    else:
        print(f"No checkpoint provided or file does not exist. Starting training from scratch, for {epochs} epochs.")

    #visualize model in console
    #print(model)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []
    val_accs = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


    for epoch in range(epochs):
        epoch_index_string = str(start_epoch + epoch + 1)
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for images, labels in track(train_loader, description=f"Epoch {epoch_index_string}"):
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
        print(f"Epoch {epoch_index_string} has Training Loss = {avg_train_loss:.4f}")
    
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
        print(f"Validation Loss: {avg_val_loss:.4f}")
        acc = correct/total*100
        val_accs.append(acc)
        print(f"Validation Accuracy: {acc:.4f}%")

        # adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # save model after each epoch
        torch.save(model.state_dict(), f"{cnn_name}_cnn/{cnn_name}_resnet18_epoch{epoch_index_string}.pth")
        
        save_for_plot(avg_train_loss, avg_val_loss, acc, epoch_index_string)
        
        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"Epoch {epoch_index_string} took {h:02}:{m:02}:{s:.2f}")

    # save model after training
    if epochs > 0:
        torch.save(model.state_dict(), f"{cnn_name}_cnn/{cnn_name}_cnn_model.pth")

    draw_plot(f"{cnn_name}_cnn/training_metrics.csv", f"{cnn_name}_cnn/training_metrics.png")

if __name__ == "__main__":
    main()