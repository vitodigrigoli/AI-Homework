import numpy as np

class Enviroment():

	def __init__(self):
		self.locationCondition = {'A': 'Dirty', 'B': 'Dirty'}

	def perceive(self, location):
		return self.locationCondition[location]
	
	def clean(self, location):
		self.locationCondition[location] = 'Clean'

	def __str__(self):
		return str(self.locationCondition)


class RobotEnviroment():

	def __init__(self):

		self.grid = np.zeros((13, 16))

		self.grid[0, :] = 1
		self.grid[12, :] = 1

		self.grid[:, 0] = 1
		self.grid[:, 15] = 1

		self.grid[1:5, 12:15] = 1
		self.grid[8:12, 12:15] = 1

		self.grid[10:12, 6:9] = 1

		self.grid[5:8, 3:9] = 1
		self.grid[6:8, 5:7] = 0

		self.robot = {'x': 2, 'y': 6}


	def __str__(self):
		s = ''
		for i in range(self.grid.shape[0]):
			for j in range(self.grid.shape[1]):
				if (i, j) == tuple(self.robot.values()):
					s += ' R ' 
				elif self.grid[i, j] == 1:
					s +='#| '  # X rappresenta un ostacolo
				else:
					s += ' . '  # . rappresenta uno spazio libero
			s +='\n'

		return s

env = RobotEnviroment()
print(env)