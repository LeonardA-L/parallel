CFLAGS=$(CCONFIG)
CFLAGS+=`pkg-config --cflags opencv`

LFLAGS=$(LCONFIG) `pkg-config opencv --libs` -lstdc++

ex1:	ex1.cu
		nvcc  $(CFLAGS) -o ex1 ex1.cu $(LFLAGS)
