all: main

main: src/main.cpp src/data.cpp src/data_handler.cpp
	g++ -Iinclude -o main src/main.cpp src/data.cpp src/data_handler.cpp
