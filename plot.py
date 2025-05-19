import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(train_losses, val_losses, tokens_seen_log=None, save_path=None):

    assert len(train_losses) == len(val_losses), "Loss lists must be same length"
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, train_losses, label="Training Loss")
    ax1.plot(epochs, val_losses, label="Validation Loss", linestyle='--')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="upper right")

    if tokens_seen_log and len(tokens_seen_log) >= len(epochs):
        ax2 = ax1.twiny()

        # Downsample or interpolate tokens_seen_log
        step = len(tokens_seen_log) // len(epochs)
        tokens_per_epoch = tokens_seen_log[::step][:len(epochs)]

        ax2.set_xticks(ax1.get_xticks())
        tick_labels = [f"{t/1e6:.1f}M" for t in tokens_per_epoch[:len(ax1.get_xticks())]]
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel("Tokens Seen")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

