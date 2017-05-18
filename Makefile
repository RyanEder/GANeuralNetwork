all:
	g++ -std=c++11 -o nn.exe main.cpp neural_structure.cpp -Wall -g
fast:
	g++ -std=c++11 -o nn.exe main.cpp neural_structure.cpp -Wall -O3
clean:
	rm -rf *.o *.exe

