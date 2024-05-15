from enviroment import Enviroment

class Vacuum():

	def __init__(self, enviroment: Enviroment):
		self.enviroment = enviroment
		self.location = 'A'

	def perceive(self):
		self.enviroment.perceive(self.location)

	def get_location(self):
		return self.location

	def move(self, direction):
		if(direction == 'Right' and self.location == 'A'):
			self.location = 'B'
		elif(direction == 'Left' and self.location == 'B'):
			self.location = 'A'
		else:
			pass

	def clean(self):
		self.enviroment.clean(self.location)


class TableVacuum(Vacuum):

	def __init__(self, enviroment: Enviroment):
		super().__init__(enviroment)

	def is_all_clean(self):
		return all(status == 'Clean' for status in self.enviroment.locationCondition.values())


	def job(self):

		while not self.is_all_clean():
			print(f'location: {self.location}\tcondition: {self.enviroment.locationCondition[self.location]}')

			if(self.location == 'A' and self.enviroment.locationCondition[self.location] == 'Clean'):
				self.move('Right')
			elif(self.location == 'A' and self.enviroment.locationCondition[self.location]  == 'Dirty'):
				self.clean()
			elif(self.location == 'B' and self.enviroment.locationCondition[self.location]  == 'Clean'):
				self.move('Left')
			elif(self.location == 'B' and self.enviroment.locationCondition[self.location]  == 'Dirty'):
				self.clean()








	