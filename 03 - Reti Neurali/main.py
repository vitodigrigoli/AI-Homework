from ToyNet import *
from utilis import *
import matplotlib.pyplot as plt

DATA = './dataset.dat'

def train_percetron(x, y, net: ToyNet, show_plot=True):
	losses = []

	while True:
		# indices = torch.randperm(len(x))
		# x = x[indices]
		# y = y[indices]

		loss = 0

		for i in range(len(x)):
			net.train_sample_percetron(x[i], y[i])
			loss += (y[i] - net.forward(x[i])).detach()**2

			if show_plot:
				plot(x, y, net, 'Perceptron Algorithm')

			print(loss.detach().item())
			losses.append(loss.detach().item())

			y_hat= torch.sign(net.forward(x))

			if torch.all(y_hat.flatten() == y):
				break

		return losses
	

def main():
	x, y = load_data(DATA)


	# simple percetron
	net = ToyNet()
	losses = train_percetron(x, y, net, False)

	print("Perceptron Algorithm")
	print("w: ", net.fc1.weight.data)
	print("b: ", net.fc1.bias.data)


	# Manual SGD
	net = ToyNet()
	losses = net.manual_SGD_train(x, y, epochs=40, lr=0.05)
	print("Manual SGD Algorithm")
	print("w: ", net.fc1.weight.data)
	print("b: ", net.fc1.bias.data)
	print('losses: ', losses)

	# PyTorch SGD
	net = ToyNet()
	losses = net.train(x, y, epochs=40, lr=0.05)
	print("PyTorch SGD Algorithm")
	print("w: ", net.fc1.weight.data)
	print("b: ", net.fc1.bias.data)
	print('losses: ', losses)




if __name__ == '__main__':
	main()