CC = g++
LD = g++
NVCC = nvcc

WARNGCC= -Wno-sign-compare -Wno-reorder -Wno-unknown-pragmas -Wno-overloaded-virtual

# --- With optimisation
CPPFLAGS = -std=c++0x -DNDEBUG -O3 -msse2 -Wall $(WARNGCC) -fopenmp
LDFLAGS = -DNEBUG -O3 -msse2 -L/usr/local/cuda-7.5/lib64/ -lcudart

# --- Debugging
#CPPFLAGS = -std=c++0x -g -Wall $(WARNGCC) 
#LDFLAGS = 


INCLUDE_DIR = -I/usr/local/cuda-7.5/include
LIB_DIR =
LIBS = `pkg-config --libs opencv` -fopenmp

simple:	sf1_cpu lab2rgb


testcpu:
	./sf1_cpu simple-data/config.txt 6 simple-data/tree

%.o: %.cpp 
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

%.o: %.cu
	$(NVCC) -c $< -o $@

main_test_simple.o: main_test_simple.cpp
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

sf1_cpu: ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o GPU.o main_test_simple.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

lab2rgb: lab2rgb.o label.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

clean:
	rm -f *.o sf1_cpu lab2rgb /tmp/features/*

