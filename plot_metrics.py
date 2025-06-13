import os
from matplotlib import pyplot as plt
import pandas as pd


def save_for_plot(train_loss, val_loss, val_acc, epoch, csv_path):
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))
    
    df = pd.DataFrame({
        'epoch': [epoch],
        'train_loss': [train_loss],
        'val_loss': [val_loss],
        'val_acc': [val_acc]
    })
    
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def draw_plot(csv_path, plot_path):
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist. Cannot draw plot.")
        return
    
    df = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # left: loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:blue')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y')

    # right: accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.plot(df['epoch'], df['val_acc']*100, label='Validation Accuracy', color='tab:green')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y')

    # legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Training Metrics')
    fig.tight_layout()
    plt.savefig(plot_path)
    plt.show()