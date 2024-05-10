import torch
import torch.nn as nn
from utils import plot, plt

class Net(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(1, 1)

	def forward(self, x):
		return self.fc1(x)

	def train(self, x, y, epochs=41, lr=0.05, show_plot = True):

		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(self.parameters(), lr=lr)

		for epoch in range(epochs):

			optimizer.zero_grad()
			y_hat = self.forward(x).flatten()

			loss = loss_fn(y_hat, y)
			loss.backward()
			optimizer.step()

			if show_plot and epoch % 10 == 0:
				plot(x, y, self , f"Epoch {epoch}, loss {loss.item():.2f}", pause=False)

		return loss.item()


class WideNet(nn.Module):

	def __init__(self, hidden_size):
		super().__init__()
		hidden_size = max(1, hidden_size)

		self.fc1 = nn.Linear(1, hidden_size)
		self.fc2 = nn.Linear(hidden_size, 1)

		self.activation = nn.ReLU()


	def forward(self, x):
		x = self.activation(self.fc1(x))
		x = self.fc2(x)

		return x
	
	def train(self, x, y, epochs=300, lr=0.015):
		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(self.parameters(), lr=lr)

		for epoch in range(epochs):
			optimizer.zero_grad()
			y_hat = self.forward(x).flatten()

			loss = loss_fn(y_hat, y)
			loss.backward()
			optimizer.step()

		return loss.item()


class DeepNet(nn.Module):

	def __init__(self, depth, hidden_size):
		super().__init__()
		hidden_size = max(1, hidden_size)  # Assicura che la dimensione dello strato nascosto sia almeno 1
		depth = max(1, depth)  # Assicura che la profondità sia almeno 1

		# Crea una lista di strati nascosti usando nn.ModuleList
		self.layers = nn.ModuleList()

		# Aggiungi il primo strato che trasforma l'input dimensione 1 in hidden_size
		self.layers.append(nn.Linear(1, hidden_size))

		# Aggiungi strati intermedi tutti con la stessa dimensione hidden_size
		for _ in range(1, depth - 1):
			self.layers.append(nn.Linear(hidden_size, hidden_size))

		# Aggiungi l'ultimo strato che trasforma l'ultimo hidden_size in output dimensione 1
		self.layers.append(nn.Linear(hidden_size, 1))
    
	def forward(self, x):
		# Applica ciascun strato nella lista ModuleList
		for layer in self.layers[:-1]:
			x = torch.relu(layer(x))  # Applica ReLU dopo ciascuno strato tranne l'ultimo
		x = self.layers[-1](x)  # L'output del modello viene generato dall'ultimo strato
		return x

	def train(self, x, y, epochs=300, lr=0.015):
		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(self.parameters(), lr=lr)

		for epoch in range(epochs):
			optimizer.zero_grad()

			y_hat = self.forward(x).flatten()
			loss = loss_fn(y_hat, y)

			loss.backward()
			optimizer.step()

		return loss.item()




