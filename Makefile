all:
	g++ -std=c++17  -I/usr/include/gtest -L/usr/lib/x86_64-linux-gnu src/test.cpp -o program.out -lgtest 

	:WARNINGS = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wformat=2\
 -Winit-self -Wmissing-declarations -Wredundant-decls -Wshadow\
 -Wstrict-overflow=5 -Wswitch-default -Wundef

FLAGS = $(WARNINGS) -std=c++11

SRC = src/test.cpp

all: 
	g++ $(FLAGS) -Ofast $(SRC) -I/usr/include/gtest -L/usr/lib/x86_64-linux-gnu include -o program.out -lgtest
	./test

debug:
	g++ $(FLAGS) -DDEBUG $(SRC) -o main
	./test