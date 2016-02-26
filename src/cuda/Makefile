NVCC=nvcc
GCC=icpc
CFLAGS=-std=c++0x -lstdc++ -O3
OFLAGS=-c
OBJ=cuda_step.o
DEP=cuda_step.cu

INC=-I$(EIGEN3_INCLUDE_DIR) -I$(GLEW_DIR)/include -I$(GLFW_DIR)/include -I/usr/local
LIB=-lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm 

all: $(OBJ)
	$(GCC) $(INC) $(CFLAGS) $(LIB) -lcudart $(OBJ) *.cpp -o Lab09_cuda

$(OBJ): cuda_step.cu
	$(NVCC) $(OFLAGS) *.cu

clean:
	rm -f *~ *.o a.out
