CC=icc
CXX=icpc
CFLAGS=-std=c++0x -Wall -pedantic -g -O3 -debug inline-debug-info
ifdef MIC
INC=-I$(EIGEN3_INCLUDE_DIR) -I/CPE419/glew-1.13.0/include -I$(GLFW_DIR)/include
else
INC=-I$(EIGEN3_INCLUDE_DIR) -I$(GLEW_DIR)/include -I$(GLFW_DIR)/include
endif

LIB=$(GLEW_DIR)/lib/libGLEW.a $(GLFW_DIR)/release/src/libglfw3.a -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -fopenmp
all:
	$(CXX) $(CFLAGS) $(INC) ../src/*.cpp $(LIB) -o Lab09

clean:
	rm -f *~ *.o a.out
