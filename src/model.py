import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import logging
import os
import sys
from torch.utils import data as utils_data
from torchvision import datasets, transforms

logging.basicConfig(filename='model.log', level=logging.INFO) # This is used to 

loss_fn = nn.CrossEntropyLoss()

class Model(nn.Module):
        def __init__(self, batch_size, lr, epochs):
                super().__init__()
                logging.info("Initializing the model")

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

                self.test_dataset = torchvision.datasets.MNIST(
                        root='./data',      # Directory to store the data
                        train=False,        # Specifies test dataset
                        download=True,      # Downloads the data if not already present
                        transform=self.my_transformer # Apply the defined transformations
                )

                self.train_dataset = datasets.MNIST(
                        root='./data',
                        train=True,
                        transform=self.my_transformer, 
                        download=True
                )
            
                self.train_dataloader = utils_data.DataLoader(
                        self.train_dataset, 
                        batch_size=self.batch_size,
                        shuffle=True
                )


                logging.info("Model initialized")

        def train(self):
                self.model.train()
                logging.info("Starting training")

                for epoch in range(self.epochs):
                        for batch_idx, (data, target) in enumerate(self.train_dataloader):
                                data = data.flatten(1)  # Flatten the images
                                self.optimizer.zero_grad()
                                output = self.forward(data)
                                loss = loss_fn(output, target)
                                loss.backward()
                                self.optimizer.step()
                                if batch_idx % 100 == 0:
                                        logging.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")
                                        print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

                logging.info("Training finished")
                torch.save(self.model.state_dict(), "model_data\\model_data.pt")


        def eval(self, index: int):
                try: 
                       self.model.load_state_dict( torch.load("model_data\\model_data.pt") )
                except Exception as exc:
                       print(f"Error: {exc}")

                try:
                    logging.info(f"Evaluating model on test dataset index: {index}")
                    self.model.eval()  # Set the model to evaluation mode

                    # Get the single image and its label from the test dataset
                    image, label = self.test_dataset[index]

                    # Reshape the image to match what the model expects:
                    # 1. image.unsqueeze(0) adds a batch dimension
                    # 2. .flatten(1) flattens the image into a 1D vector
                    image_tensor = image.unsqueeze(0).flatten(1)

                    # Use torch.no_grad() to disable gradient calculations, which is more efficient for inference
                    with torch.no_grad():
                        output = self.model(image_tensor)
                        # Get the predicted class by finding the index of the max logit
                        pred = output.argmax(dim=1)

                    print(f"\nImage at index {index}:")
                    print(f"  - True Label: {label}")
                    print(f"  - Predicted Label: {pred.item()}")

                except IndexError:
                    logging.error(f"Index {index} is out of bounds for the test dataset.")
        
        def forward(self, x):
                return self.model(x)

if __name__ == "__main__":
        logging.info("Starting script")
        my_model = Model(batch_size=256, lr=0.01, epochs=1) # Reduced epochs for faster testing

        if sys.argv[1].lower() == "train":
                my_model.train()
                sys.exit(0)
        if sys.argv[1].lower() == "eval":
                
                my_model.eval(int(sys.argv[2]))
                sys.exit(0)
               
        logging.info("Script finished")
        sys.exit(0)
