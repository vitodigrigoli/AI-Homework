import torch
import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)
        
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.softmax(self.fc3(x))
        
        return x
    
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float)
        with torch.no_grad():
            y = self.forward(x)
            print(y)
            y = y.argmax(dim=1).numpy()
            print(y)
            return y
    
    def train(self, x, y, epochs=301, lr=0.01):
        
        optimizer = torch.optim.SGD(self.parameters(), lr)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            
            y_hat = self.forward(x)
            loss = loss_fn(y_hat, y)
            
            if epoch % 50 == 0:
                print(f'Epoch: {epoch}\tloss: {loss}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        return loss
            
        
        
        
    
    
    
    