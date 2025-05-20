import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(train_losses, val_losses, tokens_seen_log=None, save_path=None):
    
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, train_losses, label="Training Loss")
    ax1.plot(epochs, val_losses, label="Validation Loss", linestyle='--')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen_log, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

