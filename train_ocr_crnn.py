import os
import string
import sys
import time
import re
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
from rich.progress import track
from plot_metrics import save_for_plot, draw_plot

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Hyperparameters
BATCH_SIZE = 32
learning_rate = 0.001

IMG_WIDTH = 280
IMG_HEIGHT = 60

CHARS = string.ascii_letters+"-.'"
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARS)}  # ctc_blank is 0
ID2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # remember ctc_blank

# encode text to indices
def encode_text(text):
    return [CHAR2ID[c] for c in text if c in CHAR2ID]

# decode model logits to text
def decode_output(logits):
    # logits: [T, B, C] → greedy decoding (naiv)
    output = logits.argmax(dim=2).transpose(0, 1)  # [B, T]
    decoded = []
    for seq in output:
        prev = -1
        chars = []
        for i in seq:
            if i.item() != prev and i.item() != 0:  # skip ctc_blank
                chars.append(ID2CHAR.get(i.item(), '?'))
            prev = i.item()
        decoded.append(''.join(chars))
    return decoded

# crnn model
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),  # 60x280 -> 30x140
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # 30x140 -> 15x70
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # height halfed, width same
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1,), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1)), # height halfed
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU()
        )

        self.map_to_seq = nn.AdaptiveAvgPool2d((1, None))  # adapt width to sequence length

        self.rnn1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        
        self.transcription = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, C=3, H=60, W=280]
        conv = self.cnn(x)  # -> [B, C=512, H', W']

        mapped_conv = self.map_to_seq(conv)  # -> [B, C=512, W'] where W' is adaptive to sequence length
        rnn_in = mapped_conv.squeeze(2).permute(0, 2, 1)  # -> [B, W=140 , C=512]

        rnn_out, _ = self.rnn1(rnn_in)  # -> [B, W=140 , C=512]
        rnn_out, _ = self.rnn2(rnn_out)  # -> [B, W=140 , C=512]

        output = self.transcription(rnn_out)  # -> [B, W=140 , num_classes]
        return output.permute(1, 0, 2)  # -> [W=140 , B, C=num_classes]
    

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
        label = row['label']
        label_encoded = torch.tensor(encode_text(label), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label_encoded

def collate_fn(batch):
    images, labels = zip(*batch)
    # pad labels to max length in batch
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    # stack images and convert to tensor
    images = torch.stack(images, dim=0)
    return images, padded_labels, label_lengths

def main():
    if len (sys.argv) > 3:
        print("Usage: python train_ocr_crnn.py [epochs=10] [model_checkpoint_epochX.pth]")
        exit()
    
    epochs = 10  # default epochs
    checkpoint_path_position = 2

    if len(sys.argv) >= 2:
        if sys.argv[1].isdigit():
            epochs = int(sys.argv[1])
        else:
            checkpoint_path_position = 1
            print(f"Using default epochs: {epochs}")
    
    
    checkpoint_path = f"setname_crnn/{sys.argv[checkpoint_path_position]}" if len(sys.argv) == checkpoint_path_position+1 else None
    print(f"Training CNN for setname_crnn with checkpoint: {checkpoint_path if checkpoint_path else 'None'}")
    
    # image transformations
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    # load dataset
    dataset = ImageDataset(
        csv_file="training_data/setname/setname_labels.csv",
        image_dir="training_data/setname",
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # alter ResNet-18 model to small image input
    model = CRNN(num_classes=NUM_CLASSES)
    # print model summary
    #print(model)

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


    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []
    val_accs = []

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


    for epoch in range(epochs):
        epoch_index_string = str(start_epoch + epoch + 1)
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for batch in track(loader, description=f"Epoch {epoch_index_string}"):
            images, labels, label_lengths = batch
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)  # T, B

            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            # accumulate training loss
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch_index_string} has Training Loss = {avg_train_loss:.4f}")
    
        # evaluation
        model.eval()
        correct = 0
        total = 0
        total_val_loss = 0
        with torch.no_grad():
            for images, labels, label_lengths in track(loader, description=f"Validation {epoch_index_string}"):
                images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)

                outputs = model(images) 
                input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)
                loss = criterion(outputs, labels, input_lengths, label_lengths)
                # accumulate validation loss
                total_val_loss +=loss.item()

                # decode outputs
                decoded = decode_output(outputs)
                # decode targets
                targets = []
                for label, length in zip(labels, label_lengths):
                    text = ''.join([ID2CHAR[idx.item()] for idx in label[:length] if idx.item() in ID2CHAR])
                    targets.append(text)

                for predicted, target in zip(decoded, targets):
                    if predicted.strip() == target.strip():
                        correct += 1
                
                total += len(targets)

        avg_val_loss = total_val_loss/len(loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        acc = correct/total
        val_accs.append(acc)
        print(f"Validation Accuracy: {acc*100:.4f}%")

        # adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # save model after each epoch
        torch.save(model.state_dict(), f"setname_crnn/setname_crnn_epoch{epoch_index_string}.pth")
        
        save_for_plot(avg_train_loss, avg_val_loss, acc, epoch_index_string)

        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"Epoch {epoch_index_string} took {h:02}:{m:02}:{s:.2f}")

    # save model after training
    if epochs > 0:
        torch.save(model.state_dict(), f"setname_crnn/setname_crnn_model.pth")



if __name__ == "__main__":
    #main()
    draw_plot("setname_crnn/training_metrics.csv", "setname_crnn/training_metrics.png")