import os
import matplotlib.pyplot as plt


def plot_loss_and_accuracy(history):
    # Plot training and validation loss and accuracy
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.plot(epochs, loss_values, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss_values, 'r', label='Validation loss')
    ax1.set_title('Training and Validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, acc_values, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_acc_values, 'r', label='Validation accuracy')
    ax2.set_title('Training and Validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    plt.show()


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    directory = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'image'
    )
    if not os.path.isdir(directory):
        os.mkdir(directory)
    fig_path = os.path.join(directory, fig_id + '.' + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)
