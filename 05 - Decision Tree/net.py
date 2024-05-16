import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		# Definizione degli strati della rete
		self.linear = nn.Linear(2, 50)  # Primo strato lineare da 2 a 50 nodi
		self.linear2 = nn.Linear(50, 50)  # Secondo strato lineare da 50 a 50 nodi
		self.linear3 = nn.Linear(50, 2)  # Terzo strato lineare da 50 a 2 nodi, per la classificazione
		self.activation = nn.ReLU()  # Funzione di attivazione ReLU
		self.softmax = nn.Softmax(dim=1)  # Softmax per le probabilità di classificazione

	def forward(self, x):
		# Definisce come i dati passano attraverso la rete
		x = self.activation(self.linear(x))  # Applica il primo strato e ReLU
		x = self.activation(self.linear2(x))  # Applica il secondo strato e ReLU
		x = self.linear3(x)  # Applica il terzo strato
		x = self.softmax(x)  # Applica Softmax
		return x

	def predict(self, x):
		# Metodo per predire la classe di input x
		x = torch.tensor(x, dtype=torch.float)
		with torch.no_grad():  # Disabilita il calcolo dei gradienti
			return self.forward(x).argmax(dim=1).numpy()  # Restituisce la classe più probabile

	def train_loop(self, X, y, epochs=900, lr=0.1):
		# Ciclo di addestramento personalizzato
		optimizer = torch.optim.SGD(self.parameters(), lr=lr)  # Usa SGD come ottimizzatore
		loss_fn = nn.CrossEntropyLoss()  # Usa CrossEntropyLoss per la classificazione
		for epoch in range(epochs):
			y_pred = self.forward(X)
			loss = loss_fn(y_pred, y)  # Calcola la perdita
			if epoch % 100 == 0:
				print(f"Epoch: {epoch}, Loss: {loss.item()}")  # Stampa la perdita ogni 100 epoche
			optimizer.zero_grad()  # Azzera i gradienti
			loss.backward()  # Calcola i gradienti
			optimizer.step()  # Aggiorna i parametri