import os
import pandas as pd
import matplotlib.pyplot as plt

def build_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def plot_loss(history, save_dir):
    fig = plt.figure(figsize=(10,3))
    fig = plt.plot(history['train_loss'], label = 'Train Loss')
    fig = plt.plot(history['val_loss'], label = 'Val Loss')
    _ = plt.legend()
    _ = plt.gca().set(title='Loss', xlabel='Epochs', ylabel='Loss')
    plt.savefig(save_dir+"/loss_curve.png")
    plt.close()

def plot_imgs(imgs, label, y, save_dir, nrows = 8, ncols = 4):
    fig, axs = plt.subplots(nrows, ncols, figsize=(14,30))
    for row in range(nrows):
        for col in range(ncols):
            axs[row][col].imshow(imgs[row*ncols + col])
            axs[row][col].set_title('Pred : '+str(label[row*ncols + col])+'\n Act : '+str(y[row*ncols + col]))
            axs[row][col].set(xticks=[], yticks=[])
    plt.savefig(save_dir+'/predictions.png')
    plt.close()

def log_metrics(history, test_loss, test_acc, save_dir) :
    epochs = len( history['train_loss'])
    
    train_loss = history['train_loss'][-1]
    train_acc = history['train_acc'][-1]

    val_loss = history['val_loss'][-1]
    val_acc = history['val_acc'][-1]

    df = pd.DataFrame.from_dict({
        'Dataset' : ['Train', 'Val', 'Test'], 
        'Loss' : [train_loss, val_loss, test_loss],
        'Acc' : [train_acc, val_acc, test_acc]
    })
    df.to_csv(save_dir+f'/metrics_epochs_{epochs}.csv', index=False)

