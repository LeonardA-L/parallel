CFLAGS=$(CCONFIG)
CFLAGS+=`pkg-config --cflags opencv`

LFLAGS=$(LCONFIG) `pkg-config opencv --libs` -lstdc++

sample:	sample.cu
	nvcc  $(CFLAGS) -o sample sample.cu $(LFLAGS)
