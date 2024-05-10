from nets import *
from utils import *

FILE = './dataset2.dat'

def simpleNet():

	x, y = load_data(FILE)
	x_train, y_train = x[:len(x)//2], y[:len(y)//2]
	x_test, y_test = x[len(x)//2:], y[len(y)//2:]

	net = Net()
	train_loss = net.train(x_train, y_train)

	test_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
	plot(x_test , y_test , net , f"Train loss {train_loss :.2f}, val loss {test_loss :.2f}")



def wideNet():
	x, y = load_data(FILE)

	x_train, y_train = x[:len(x)//2], y[:len(y)//2]
	x_test, y_test = x[len(x)//2:], y[len(y)//2:]

	for i in range (0, 101, 20):
		net = WideNet(i)

		pytorch_total_params = sum(p.numel() for p in net.parameters () if p.requires_grad)
		train_loss = net.train(x_train, y_train)
		val_loss = nn.MSELoss()(net(x_test).squeeze (), y_test)

		plot(x_test , y_test , net , f"Hidden size {i}, params {pytorch_total_params}\ntrain loss {train_loss :.2f}, val loss {val_loss :.2f}")


def deepNet():
	x, y = load_data(FILE)
	x_train, y_train = x[:len(x)//2], y[:len(y)//2]
	x_test, y_test = x[len(x)//2:], y[len(x)//2:]

	for i in range(3, 19, 3):
		net = DeepNet(hidden_size=300, depth=i)
		pytorch_total_params = sum(p.numel() for p in net.parameters () if p.requires_grad)
		train_loss = net.train(x_train, y_train)
		val_loss = nn.MSELoss()(net(x_test).squeeze (), y_test)

		plot(x_test , y_test , net , f"Depth {i}, params {pytorch_total_params}\ntrain loss {train_loss :.2f}, val loss {val_loss :.2f}")



	


if __name__ == '__main__':
	deepNet()