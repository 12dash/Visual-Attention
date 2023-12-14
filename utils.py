import os
import matplotlib.pyplot as plt

def build_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def plot_loss(history, save_dir):
    fig = plt.figure(figsize=(10,3))
    fig = plt.plot(history['loss'], label = 'Train Loss')
    fig = plt.plot(history['val_loss'], label = 'Val Loss')
    _ = plt.legend()
    _ = plt.gca().set(title='Loss', xlabel='Epochs', ylabel='Loss')
    plt.savefig(save_dir+"loss_curve.png")
    plt.close()
