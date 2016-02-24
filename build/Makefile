CC=icc
CXX=icpc
CFLAGS=-std=c++0x -Wall -pedantic -O3
INC=-I$(EIGEN3_INCLUDE_DIR) -I$(GLEW_DIR)/include -I$(GLFW_DIR)/include
LIB=$(GLEW_DIR)/lib/libGLEW.a $(GLFW_DIR)/release/src/libglfw3.a -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -fopenmp
all:
	$(CXX) $(CFLAGS) $(INC) ../src/*.cpp $(LIB) -o Lab09

clean:
	rm -f *~ *.o a.out
