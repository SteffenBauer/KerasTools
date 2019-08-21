#!/usr/bin/env python3

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def update_progress(msg, progress):
    barLength = 40
    status = ""
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2:.2%} {3}".format(msg, "="*(block-1) + ">" + "-"*(barLength-block), progress, status)
    sys.stdout.write(text)
    sys.stdout.flush()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            update_progress('Train Epoch: {} [{}/{} ] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                loss.item()), batch_idx / len(train_loader))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

no_cuda = False
seed = 1
batch_size = 64
epochs = 10

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 6, 'pin_memory': False} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/.cache/torch/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/.cache/torch/data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

start_time = time.time()
model = Net().to(device)
print(model)
end_time = time.time()
print("Build time", end_time - start_time)

start_time = time.time()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
end_time = time.time()
print("Compile time", end_time - start_time)

start_time = time.time()
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    print()
end_time = time.time()
print("Train time", end_time - start_time)

start_time = time.time()
test(model, device, test_loader)
end_time = time.time()
print("Test time", end_time - start_time)

