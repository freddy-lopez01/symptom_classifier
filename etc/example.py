def readfile(filename):
	with open(filename, "r") as myfile:
		for everyline in myfile:
			print(everyline)




def main():
	x = input("input file you want to read: ")
	readfile(x)

if __name__ == "__main__":
	main()
