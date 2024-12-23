import torch
import torch.nn as nn
import torch.optim as optim
import struct
import numpy as np
import time
import math
# Constants matching the C implementation
INPUT_LAYER_SIZE = 28 * 28
HIDDEN_LAYER_SIZE = 256
OUTPUT_LAYER_SIZE = 10
LEARNING_RATE = 0.0005
EPOCHS = 10
TRAIN_SPLIT = 0.8


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        self.hidden = nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.output = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
        
        # Match He initialization from C code
        for layer in [self.hidden, self.output]:
            scale = math.sqrt(2.0 / layer.in_features)
            nn.init.uniform_(layer.weight, -scale, scale)
            nn.init.zeros_(layer.bias)
            
    def forward(self, x):
        x = x.view(-1, INPUT_LAYER_SIZE)
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return torch.softmax(x, dim=1)

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        # Read image data and make array writeable
        images = np.frombuffer(f.read(), dtype=np.uint8).copy()
        images = images.reshape(num_images, rows * cols)
        return images

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        # Read magic number and number of items
        magic = struct.unpack('>I', f.read(4))[0]
        num_items = struct.unpack('>I', f.read(4))[0]
        
        # Read labels and make array writeable
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
        return labels

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read data from existing files
    images = read_idx_images('./data/train-images.idx3-ubyte')
    labels = read_idx_labels('./data/train-labels.idx1-ubyte')
    
    # Convert to tensors and normalize
    images = torch.from_numpy(images).float() / 255.0
    labels = torch.from_numpy(labels).long()
    
    train_size = int(len(images) * TRAIN_SPLIT)
    test_size = len(images) - train_size
    
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        # Training loop - sample by sample like C implementation
        for i in range(train_size):
            data = images[i].to(device)
            target = labels[i].to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Testing loop - sample by sample
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(train_size, len(images)):
                data = images[i].to(device)
                target = labels[i].to(device)
                output = model(data)
                pred = output.argmax()
                correct += pred.eq(target).item()
        
        accuracy = 100. * correct / test_size
        avg_loss = total_loss / train_size
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds')

if __name__ == '__main__':
    train_model()
