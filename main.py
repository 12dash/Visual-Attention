import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model.model import AttentionModel
from dataloaders.MnistDataset import Mnist
from utils import plot_loss, build_dir


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

if __name__ == "__main__":
    base_dir = os.getcwd() + '/data/mnist'

    img_gzip = "/train-images.idx3-ubyte"
    label_gzip = "/train-labels.idx1-ubyte"

    test_img_gzip = "/t10k-images.idx3-ubyte"
    test_label_gzip = "/t10k-labels.idx1-ubyte"

    IMG_SHAPE = (28,28)
    PATCH_SIZE = 4
    NUM_PATCH = (IMG_SHAPE[0]//PATCH_SIZE) * (IMG_SHAPE[1]//PATCH_SIZE)
    BATCH_SIZE = 16
    OUTPUT_DIM = 10

    global DEVICE
    DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else DEVICE
    DEVICE = torch.device('cpu')

    model = AttentionModel(patch_size = PATCH_SIZE, output_dim = OUTPUT_DIM, L=4, num_heads=2,
                            NUM_PATCH = NUM_PATCH).to(DEVICE)

    train_dataset = Mnist(img_gzip = img_gzip, label_gzip = label_gzip, base_dir = base_dir)
    test_dataset = Mnist(img_gzip = test_img_gzip, label_gzip = test_label_gzip, base_dir = base_dir)

    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 20
    history = {'loss':[]}
    for epoch in tqdm(range(epochs)):
        acc = 0
        loss, acc = train_step(model, train_dataloader, loss_fn, optimizer)
        history['loss'].append(loss)
        test_loss, test_acc = evaluate(model, test_dataloader, loss_fn)

        print(f"[{epoch+1}] Loss : {loss:.4f} Acc : {acc : .2f} \
        Test Loss : {test_loss: .4f} Test Acc : {test_acc : .2f}")
    