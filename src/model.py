import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils as utils
import sys
from torchvision import datasets, transforms

loss_fn = nn.CrossEntropyLoss()

class Model(nn.Module):
        def __init__(self, batch_size, lr, epochs):
                super().__init__()

                self.batch_size: int = batch_size
                self.epochs: int = epochs

                self.my_transformer = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                ])

                self.model = nn.Sequential(
                        nn.Linear(28*28, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                )

                self.optimizer = optim.SGD(self.parameters(), lr=lr)

                self.dataset = datasets.MNIST(root='./data', train=True, transform=self.my_transformer, download=True)
                self.dataloader = utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        def train(self):
                self.model.train()

                for epoch in range(self.epochs):
                        for batch_idx, (data, target) in enumerate(self.dataloader):
                                data = data.flatten(1)  # Flatten the images
                                self.optimizer.zero_grad()
                                output = self.forward(data)
                                loss = loss_fn(output, target)
                                loss.backward()
                                self.optimizer.step()
                                if batch_idx % 100 == 0:
                                        print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

        def __call__(self, x):
                return self.forward(x)

        def forward(self, x):
                return self.model(x)

if __name__ == "__main__":
        my_model = Model(batch_size=128, lr=0.01, epochs=20)
        my_model.train()