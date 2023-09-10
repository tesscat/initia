run:
	syclcc -I vendor/eigen/ -I vendor/OpenSYCL/include/ -I src src/main.cpp -g
