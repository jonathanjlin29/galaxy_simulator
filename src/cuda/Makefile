NVCC=nvcc
GCC=gcc
CFLAGS=-std=c++0x -O3
OFLAGS=-c
OBJ=cuda_step.o
DEP=cuda_step.cu

INC=-I$(EIGEN3_INCLUDE_DIR) -I$(GLEW_DIR)/include -I$(GLFW_DIR)/include
LIB=-lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -fopenmp

all: $(OBJ)
	$(GCC) $(INC) $(CFLAGS) $(LIB) -lcudart $(OBJ) *.cpp -o Lab09_cuda

$(OBJ): cuda_step.cu
	$(NVCC) $(OFLAGS) *.cu


clean:
	rm -f *~ *.o a.out
