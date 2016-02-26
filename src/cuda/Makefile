CC=icc
NVCC=nvcc
CXX=gcc
CFLAGS=-std=c++0x -pedantic -O3 -lstdc++
OFLAGS=-c
OBJ=cuda_step.o
DEP=cuda_step.cu
ifdef MIC
INC=-I$(EIGEN3_INCLUDE_DIR) -I/CPE419/glew-1.13.0/include -I$(GLFW_DIR)/include
else
INC=-I$(EIGEN3_INCLUDE_DIR) -I$(GLEW_DIR)/include -I$(GLFW_DIR)/include
endif

LIB=$(GLEW_DIR)/lib/libGLEW.a $(GLFW_DIR)/release/src/libglfw3.a -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -lcudart
all:$(OBJ)
	$(CXX) $(CFLAGS) $(INC) *.cpp cuda_step.o $(LIB) -o Lab09

$(OBJ): cuda_step.cu
	$(NVCC) $(OFLAGS) *.cu

clean:
	rm -f *~ *.o a.out
