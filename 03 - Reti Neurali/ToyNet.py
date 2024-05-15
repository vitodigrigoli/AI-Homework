import torch
import torch.nn as nn
from utilis import plot

class ToyNet(nn.Module):

	def __init__(self):
		super(ToyNet, self).__init__()
		self.fc1 = nn.Linear(2, 1)


	def forward(self, x):
		return self.fc1(x)
	

	def train_sample_percetron(self, x, y):
		y_hat = torch.sign(self.forward(x))

		delta = (y - y_hat) / 2
		self.fc1.weight.data += delta * x
		self.fc1.bias.data += delta
	

	def manual_SGD_train(self, x, y, epochs=50, lr=0.05, showPlot = True):
		losses = []

		for _ in range(epochs):
			loss = 0

			for i in range(len(x)):
				y_hat = self.forward(x[i])
				error = y_hat - y[i]

				self.fc1.weight.data -= lr * error * x[i]
				self.fc1.bias.data  -= lr * error * 1
				
				loss += (error.detach()**2)

			losses.append(loss.item())

			if showPlot:
				plot(x, y, self, title="Manual SGD", temp=False)

			return losses


	def train(self, x, y, epochs=40, lr=0.05, showPlot=True):
		
		##################################################################
		# Inizializzazione dell'Ottimizzatore e della Funzione di Perdita:
		##################################################################

		# Crea un oggetto ottimizzatore SGD che aggiornerà i parametri (self.parameters()) 
		# del modello con un learning rate specificato (lr)
		optimizer = torch.optim.SGD(self.parameters(), lr=lr)

		#Definisce la funzione di perdita come il Mean Squared Error, comune per problemi di regressione.
		criterion = nn.MSELoss()

		losses = []

		################################################################
		# Loop di Addestramento per Epoca:
		################################################################

		for _ in range(epochs):

			################################################################
			# Forward Pass e Calcolo della Perdita
			################################################################

			# Prima di calcolare il gradiente per ogni epoca, azzeriamo i gradienti esistenti
			# per evitare l'accumulo tra epoche, una pratica necessaria in PyTorch.
			optimizer.zero_grad()

			# Esegue il forward pass attraverso la rete (usando il metodo forward definito nella classe)
			# e poi appiattisce l'output per garantire che corrisponda alla forma di y.
			y_hat = self.forward(x).flatten()

			# Calcola la perdita MSE tra le predizioni y_hat e i veri target y.
			loss = criterion(y_hat, y)

			################################################################
			# Backward Pass e Aggiornamento dei Parametri
			################################################################

			# Calcola i gradienti della perdita rispetto ai parametri del modello.
			loss.backward()

			# Aggiorna i parametri del modello basandosi sui gradienti calcolati.
			optimizer.step()

			################################################################
			# Registrazione della perdita ed eventuale visualizzazione
			################################################################

			#  Salva il valore della perdita, dopo averne staccato il gradiente per impedire
			#  ulteriori calcoli con esso, in una lista di perdite per l'analisi
			losses.append(loss.detach())

		if showPlot:
			plot(x, y, self, title="PyTorch SGD Optimizer", temp=False)
		
		return losses
