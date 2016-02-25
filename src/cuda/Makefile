CC=icc
CXX=icpc
NVCC=nvcc
#CFLAGS=-std=c++0x -Wall -pedantic -O3 -lcublas
CFLAGS=-O2 -lcublas
ifdef MIC
INC=-I$(EIGEN3_INCLUDE_DIR) -I/CPE419/glew-1.13.0/include -I$(GLFW_DIR)/include
else
INC=-I$(EIGEN3_INCLUDE_DIR) -I$(GLEW_DIR)/include -I$(GLFW_DIR)/include
endif

LIB=$(GLEW_DIR)/lib/libGLEW.a $(GLFW_DIR)/release/src/libglfw3.a -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -fopenmp
all:
	$(NVCC) $(CFLAGS) $(INC) *.cpp -o Lab09_cuda

#	$(NVCC) $(CFLAGS) $(INC) ../src/*.cpp $(LIB) -o Lab09_cuda

clean:
	rm -f *~ *.o a.out
