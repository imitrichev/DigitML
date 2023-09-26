all:
		g++ -std=c++17 -I/usr/include/gtest -L/usr/lib/x86_64-linux-gnu src/test.cpp -o program.out -lgtest
