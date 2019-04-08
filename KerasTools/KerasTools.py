import sys
import matplotlib.pyplot as plt

def update_progress(msg, progress):
    barLength = 40
    status = ""
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2:.2%} {3}".format(msg, "="*(block-1) + ">" + "-"*(barLength-block), progress, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def plot_history(history):

    epochs = range(1, len(history['loss']) + 1)

    if 'test_loss' in history:
        print("Test loss:", history['test_loss'])
    if 'test_acc' in history:
        print("Test accuracy:", history['test_acc'])
    if 'baseline' in history:
        print("Baseline: ", history['baseline'])
    if 'epochs' in history:
        train_epochs = history['epochs']
        print("Final training epochs:", train_epochs)
    else:
        train_epochs = epochs

    plt.figure(figsize=(15,5))

    if 'acc' in history:
        plt.subplot(1, 2, 1)

    plt.plot(epochs, history['loss'], linewidth=2.0, label='train loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], linewidth=2.0, label='val loss')
    if 'test_loss' in history:
        plt.plot([train_epochs], history['test_loss'], 'rx', ms=10.0, label='test loss')
        plt.axvline(train_epochs, ls=':', linewidth=2.0)
    if 'baseline' in history:
        plt.axhline(history['baseline'], ls=':', linewidth=2.0, color='red', label='baseline')
    plt.ylabel('Loss')
    plt.legend()

    if 'acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['acc'], linewidth=2.0, label='train acc')
        if 'val_acc' in history:
            plt.plot(epochs, history['val_acc'], linewidth=2.0, label='val acc')
        if 'test_acc' in history:
            plt.plot([train_epochs], history['test_acc'], 'rx', ms=10.0, label='test acc')
            plt.axvline(train_epochs, ls=':', linewidth=2.0)
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()

    plt.show()


