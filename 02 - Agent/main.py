from vacuum import Enviroment, Vacuum, TableVacuum

def main():

	env = Enviroment()

	vacuum = TableVacuum(env)

	print(env)

	vacuum.job()
	print(env)



	


if __name__ == '__main__':
	main()