import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc(x))
        return x
    
    
class Decoder(nn.Module):
    
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class AEClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, input_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        encoded = self.encoder(x)
        x = self.activation(self.fc1(encoded))
        x = self.softmax(self.fc2(x))
        return x
    
    def autoencode(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def classify(self, x):
        with torch.no_grad():
            return torch.argmax(self(x), dim=1)
        
    def pretrain_autoencoder(self, train_loader, epochs=50, lr=0.01):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        losses = []
        
        for _ in range(epochs):
            for batch in train_loader:
                x, _ = batch
                optimizer.zero_grad()
                output = self.autoencode(x)
                loss = criterion(output, x)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            
        return losses
    
    def train_classifier(self, train_loader, epoches=50, lr=0.01):
        
        # pretrain autoencoder
        ae_losses = self.pretrain_autoencoder(train_loader)
        
        # freeze the encoder weight
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
            
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses = []
        
        for _ in range(epoches):
            for batch in train_loader:
                x, y = batch
                optimizer.zero_grad()
                y_hat = self.forward(x)
                loss = criterion(y_hat, y)
                
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            
        return losses
        
            
        
        
                
            
            
        
    