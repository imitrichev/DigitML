WARNINGS = -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wformat=2\
 -Winit-self -Wmissing-declarations -Wredundant-decls -Wshadow\
 -Wstrict-overflow=5 -Wswitch-default -Wundef

FLAGS = $(WARNINGS) -std=c++17 

SRC = src/main.cpp

sigmoid:
	g++ $(FLAGS) -Ofast $(SRC) -I include -o main

hyper_tan:
	g++ $(FLAGS) -DHYPER_TAN -Ofast $(SRC) -I include -o main

test:
	g++ $(FLAGS)  -lgtest  -DTEST -Ofast $(SRC) -I include -o main


all: sigmoid


debug:
	g++ $(FLAGS) -DDEBUG $(SRC) -o main

