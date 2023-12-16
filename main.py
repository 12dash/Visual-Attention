import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model.model import AttentionModel
from datasets.MnistDataset import Mnist
from datasets.CifarDataset import Cifar
from datasets.Cifar100Dataset import Cifar100
from utils import plot_loss, build_dir, plot_imgs, log_metrics


def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    global DEVICE
    loss_, acc = [],0
    for _, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        output = model(x_batch)

        # Calculate loss
        loss = loss_fn(output, y_batch)

        # Calculate accuracy
        acc += (torch.argmax(output, dim = -1) == y_batch).sum() / x_batch.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())

    acc = acc / len(dataloader)
    return np.mean(loss_), acc.item()

def evaluate(model, dataloader, loss_fn):
    global DEVICE

    model.eval()
    loss_, acc = [], 0

    for _, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        output = torch.argmax(output, dim =  -1)
        output = (output == y_batch).sum()
        acc += output.item() / x_batch.size(0)
        loss_.append(loss.item())
    
    acc = acc / len(dataloader)
    loss_ = np.mean(loss_)
    return loss_, acc

def load_mnist_dataset():
    print('Loading MNIST Dataset')
    global PATCH_SIZE
    global BATCH_SIZE
    global NUM_PATCH
    global INPUT_CHANNEL
    global SAVE_DIR
    global OUTPUT_DIM

    OUTPUT_DIM = 10
   
    base_dir = os.getcwd() + '/data/mnist'
    SAVE_DIR = 'result/mnist'

    img_gzip = "/train-images.idx3-ubyte"
    label_gzip = "/train-labels.idx1-ubyte"

    test_img_gzip = "/t10k-images.idx3-ubyte"
    test_label_gzip = "/t10k-labels.idx1-ubyte"

    IMG_SHAPE = (28,28)
    INPUT_CHANNEL = 1
    NUM_PATCH = (IMG_SHAPE[0]//PATCH_SIZE) * (IMG_SHAPE[1]//PATCH_SIZE)
    train_dataset = Mnist(img_gzip = img_gzip, label_gzip = label_gzip, base_dir = base_dir)
    test_dataset = Mnist(img_gzip = test_img_gzip, label_gzip = test_label_gzip, base_dir = base_dir)

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)

    return train_dataloader, test_dataloader, test_dataloader, train_dataset.label_name


def load_cifar10_dataset():
    print('Loading Cifar 10 Dataset')
    global PATCH_SIZE
    global BATCH_SIZE
    global NUM_PATCH
    global INPUT_CHANNEL
    global SAVE_DIR
    global OUTPUT_DIM
    
    base_dir = os.getcwd() + '/data/cifar-10'

    img_gzip = ["/data_batch_1","/data_batch_2","/data_batch_3", "/data_batch_4"]
    val_gzip = ['/data_batch_5']
    test_img_gzip = ["/test_batch"]

    label_name_zip = '/batches.meta'
    SAVE_DIR = 'result/cifar-10'

    IMG_SHAPE = (32,32)
    INPUT_CHANNEL = 3
    NUM_PATCH = (IMG_SHAPE[0]//PATCH_SIZE) * (IMG_SHAPE[1]//PATCH_SIZE)
    train_dataset = Cifar(img_gzip = img_gzip, label_name_zip = label_name_zip, base_dir = base_dir, transform=True)
    val_dataset = Cifar(img_gzip = val_gzip, label_name_zip = label_name_zip, base_dir = base_dir)
    test_dataset = Cifar(img_gzip = test_img_gzip, label_name_zip = label_name_zip, base_dir = base_dir)

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)

    OUTPUT_DIM = train_dataset.classes

    return train_dataloader, val_dataloader, test_dataloader, train_dataset.label_name

def load_cifar100_dataset():
    print('Loading Cifar 100 Dataset')
    global PATCH_SIZE
    global BATCH_SIZE
    global NUM_PATCH
    global INPUT_CHANNEL
    global SAVE_DIR
    global OUTPUT_DIM
    
    base_dir = os.getcwd() + '/data/cifar-100'

    img_zip = "/train"
    test_img_zip = "/test"
    label_name_zip = '/meta'

    SAVE_DIR = 'result/cifar-100'

    IMG_SHAPE = (32,32)
    INPUT_CHANNEL = 3
    NUM_PATCH = (IMG_SHAPE[0]//PATCH_SIZE) * (IMG_SHAPE[1]//PATCH_SIZE)

    train_dataset = Cifar100(img_gzip = img_zip, label_name_zip = label_name_zip, base_dir = base_dir, transform=True)
    val_dataset = Cifar100(img_gzip = test_img_zip, label_name_zip = label_name_zip, base_dir = base_dir)
    test_dataset = Cifar100(img_gzip = test_img_zip, label_name_zip = label_name_zip, base_dir = base_dir)

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)
    OUTPUT_DIM = train_dataset.classes

    return train_dataloader, val_dataloader, test_dataloader, train_dataset.label_name

def sample_batch(x, y, model, labels=None):
    global DEVICE
    global SAVE_DIR

    x = x.to(DEVICE)
    y = y.numpy()
    pred = torch.argmax(model(x), dim =  -1).cpu().numpy()
    if labels is not None :
        labels = np.array(labels)
        y, pred = labels[y], labels[pred]
    x = np.transpose(x.cpu(), (0, 2, 3, 1))
    plot_imgs(x, pred, y, SAVE_DIR)


if __name__ == "__main__":
    global PATCH_SIZE
    global BATCH_SIZE
    global NUM_PATCH
    global INPUT_CHANNEL
    global SAVE_DIR
    
    PATCH_SIZE = 4
    BATCH_SIZE = 32

    global DEVICE
    DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else DEVICE

    train_dataloader, val_dataloader, test_dataloader, label_name = load_mnist_dataset()
    build_dir(save_dir=SAVE_DIR)

    model = AttentionModel(patch_size = PATCH_SIZE, 
                           output_dim = OUTPUT_DIM, 
                           L=12, num_heads=8,
                           embedding_dim = 256,
                           projection_dim = 512,
                           input_channel = INPUT_CHANNEL,
                           NUM_PATCH = NUM_PATCH).to(DEVICE)
    

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    epochs = 5

    history = {'train_loss':[],  'val_loss':[], 'train_acc':[], 'val_acc' : []}
    for epoch in tqdm(range(epochs)):
        acc = 0
        loss, acc = train_step(model, train_dataloader, loss_fn, optimizer)
        history['train_loss'].append(loss)
        history['train_acc'].append(acc)

        val_loss, val_acc = evaluate(model, val_dataloader, loss_fn)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[{epoch+1}] Loss : {loss:.4f} Acc : {acc : .2f} \
        Val Loss : {val_loss: .4f} Val Acc : {val_acc : .2f}")
    
    test_loss, test_acc = evaluate(model, test_dataloader, loss_fn)
    print(f"Test Loss : {test_loss: .4f} Test Acc : {test_acc : .2f}")
    plot_loss(history, SAVE_DIR)
    
    x, y = next(iter(test_dataloader))
    sample_batch(x, y, model, labels = label_name)
    log_metrics(history, test_loss, test_acc, SAVE_DIR)

    
    