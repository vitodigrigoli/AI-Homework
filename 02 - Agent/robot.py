from enviroment import RobotEnviroment

class Robot():

	def __init__(self, env: RobotEnviroment):
		self.enviroment = env
		self.visited = set()
		self.visited.add((env.robot['x'], env.robot['y']))

	def move(self, direction):
		match direction:
			case 'up':
				self.enviroment.robot['x'] -= 1

			case 'down':
				self.enviroment.robot['x'] += 1

			case 'left':
				self.enviroment.robot['y'] -= 1

			case 'right':
				self.enviroment.robot['y'] += 1

			case _:
				print('Invalid Direction')

		self.visited.add((self.enviroment.robot['x'], self.enviroment.robot['y']))

	
	def __str__(self) -> str:
		return f"X: {self.enviroment.robot['x']}\tY: {self.enviroment.robot['y']}"
	
	def has_visited(self, x, y):
		return (x, y) in self.visited
	

	def perceive(self):
		x = self.enviroment.robot['x']
		y = self.enviroment.robot['y']
		print(x, y)

		top = self.enviroment.grid[x-1, y]
		right = self.enviroment.grid[x, y+1]
		bottom = self.enviroment.grid[x+1, y]
		left = self.enviroment.grid[x, y-1]

		return top, right, bottom, left
	

	def action(self):
		top, right, bottom, left = self.perceive()
		print(top, right, bottom, left)

		x, y = self.enviroment.robot['x'], self.enviroment.robot['y']

		if(not top and not self.has_visited(x-1, y)):
			self.move('up')
		elif(not right and not self.has_visited(x, y+1) ):
			self.move('right')
		elif(not bottom and not self.has_visited(x+1, y)):
			self.move('down')
		elif(not left and not self.has_visited(x, y-1)):
			self.move('left')
		else:
			print('Obstacles, Obstacles Eveywhere!')


def main():
	env = RobotEnviroment()
	robot = Robot(env)

	for i in range(50):
		robot.action()
		print(robot.enviroment)






if __name__ == '__main__':
	main()