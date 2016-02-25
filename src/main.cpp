#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#define GLEW_STATIC
// use this to run on mic comment out if no mic
//#include "/CPE419/glew-1.13.0/include/GL/glew.h"
// use when no mic
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Camera.h"
#include "GLSL.h"
#include "Program.h"
#include "MatrixStack.h"
#include "Particle.h"
#include "Texture.h"

using namespace std;
using namespace Eigen;

bool keyToggles[256] = {false}; // only for English keyboards!

GLFWwindow *window; // Main application window
string RESOURCE_DIR = ""; // Where the resources are loaded from

shared_ptr<Program> progSimple;
shared_ptr<Program> prog;
shared_ptr<Camera> camera;

shared_ptr<Texture> texture;

vector< shared_ptr<Particle> > particles;

//PARRALLEL ARRAYS FOR PARTICLE

vector <double> masses;

vector <double> positionx;
vector <double> positiony;
vector <double> positionz;

vector <double> velocityx;
vector <double> velocityy;
vector <double> velocityz;

vector <double> accelerx;
vector <double> accelery;
vector <double> accelerz;

vector <float> radius;

vector <float> colorx;
vector <float> colory;
vector <float> colorz;
vector <float> colorW;

vector <vector <float> > posBuf;
vector <vector <float> > texBuf;
vector <vector <float> > indBuf;

vector <GLuint> posBufID;
vector <GLuint> texBufID;
vector <GLuint> indBufID;
//PARRALLEL ARRAYS FOR PARTICLE


double t, h, e2;

static void error_callback(int error, const char *description)
{
	cerr << description << endl;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

static void char_callback(GLFWwindow *window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
}

static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if(state == GLFW_PRESS) {
		camera->mouseMoved(xmouse, ymouse);
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	if(action == GLFW_PRESS) {
		bool shift = mods & GLFW_MOD_SHIFT;
		bool ctrl  = mods & GLFW_MOD_CONTROL;
		bool alt   = mods & GLFW_MOD_ALT;
		camera->mouseClicked(xmouse, ymouse, shift, ctrl, alt);
	}
}

static void initGL()
{
	GLSL::checkVersion();

	// Set background color
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	// Enable z-buffer test
	glEnable(GL_DEPTH_TEST);
	// Enable alpha blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	progSimple = make_shared<Program>();
	progSimple->setShaderNames(RESOURCE_DIR + "simple_vert.glsl", RESOURCE_DIR + "simple_frag.glsl");
	progSimple->setVerbose(false); // Set this to true when debugging.
	progSimple->init();
	progSimple->addUniform("P");
	progSimple->addUniform("MV");

	prog = make_shared<Program>();
	prog->setVerbose(true); // Set this to true when debugging.
	prog->setShaderNames(RESOURCE_DIR + "particle_vert.glsl", RESOURCE_DIR + "particle_frag.glsl");
	prog->init();
	prog->addUniform("P");
	prog->addUniform("MV");
	prog->addAttribute("vertPos");
	prog->addAttribute("vertTex");
	prog->addUniform("radius");
	prog->addUniform("alphaTexture");
	prog->addUniform("color");

	texture = make_shared<Texture>();
	texture->setFilename(RESOURCE_DIR + "alpha.jpg");
	texture->init();

	camera = make_shared<Camera>();

	// Initialize OpenGL for particles.
	for(int i = 0; i < particles.size(); ++i) {
		particles[i]->init();
	}

	// If there were any OpenGL errors, this will print something.
	// You can intersperse this line in your code to find the exact location
	// of your OpenGL error.
	GLSL::checkError(GET_FILE_LINE);
}

// Sort particles by their z values in camera space
class ParticleSorter {
public:
	bool operator()(size_t i0, size_t i1) const
	{
//		// Particle positions in world space
//		const Vector3d &x0 = particles[i0]->getPosition();
//		const Vector3d &x1 = particles[i1]->getPosition();
	const double x0x = positionx.at(i0);
	const double x0y = positiony.at(i0);
	const double x0z = positionz.at(i0);

	const double x1x = positionx.at(i1);
	const double x1y = positiony.at(i1);
	const double x1z = positionz.at(i1);

		// Particle positions in camera space
		float z0 = V.row(2) * Vector4f(x0x, x0y, x0z, 1.0f);
		float z1 = V.row(2) * Vector4f(x1x, x1y, x1z, 1.0f);
		return z0 < z1;
	}

	Matrix4f V; // current view matrix
};
ParticleSorter sorter;

// http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
vector<size_t> sortIndices(const vector<T> &v) {
	// initialize original index locations
	vector<size_t> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(), sorter);
	return idx;
}

void renderGL()
{
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);

	// Use the window size for camera.
	glfwGetWindowSize(window, &width, &height);
	camera->setAspect((float)width/(float)height);

	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if(keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	} else {
		glDisable(GL_CULL_FACE);
	}
	if(keyToggles[(unsigned)'l']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();

	// Apply camera transforms
	P->pushMatrix();
	camera->applyProjectionMatrix(P);
	MV->pushMatrix();
	camera->applyViewMatrix(MV);
	// Set view matrix for the sorter
	sorter.V = MV->topMatrix();

	// Draw particles
	prog->bind();
	texture->bind(prog->getUniform("alphaTexture"), 0);
	glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, P->topMatrix().data());
	// Sort particles by Z for transparency rendering.
	// Since we don't want to modify the contents of the vector, we compute the
	// sorted indices and traverse the particles in this sorted order.
//	auto sortedIndices = sortIndices(particles);
//    auto sortedIndexx = sortIndicices(positionx);
//    auto sortedIndexy = sortIndicices(positiony);
    auto sortedIndexz = sortIndices(positionz);
//	for(int i = 0; i < sortedIndices.size(); ++i) {
//		int ii = sortedIndices[i];
//		particles[ii]->draw(prog, MV);
//	}


#pragma omp parallel for
    for(int i = 0; i < sortedIndexz.size(); i++) {
	int ii = sortedIndexz[i];
	/*
	 void setPosition(const Eigen::Vector3d x) { this->x = x; }
	 void setVelocity(const Eigen::Vector3d v) { this->v = v; }
	 */
	particles[ii]->setPosition(Vector3d(positionx.at(ii), positiony.at(ii), positionz.at(ii)));
	particles[ii]->setVelocity(Vector3d(velocityx.at(ii), velocityy.at(ii), velocityz.at(ii)));
	particles[ii]->draw(prog, MV);
    }
	texture->unbind(0);
	prog->unbind();

	//////////////////////////////////////////////////////
	// Cleanup
	//////////////////////////////////////////////////////

	// Pop stacks
	MV->popMatrix();
	P->popMatrix();

	GLSL::checkError(GET_FILE_LINE);
}

void saveParticles(const char *filename)
{
	ofstream out(filename);
	if(!out.good()) {
		cout << "Could not open " << filename << endl;
		return;
	}

	// 1st line:
	// <n> <h> <e2>
	out << particles.size() << " " << h << " " << " " << e2 << endl;

	// Rest of the lines:
	// <mass> <position> <velocity> <color> <radius>

	//
	// IMPLEMENT ME
	//

	out.close();
	cout << "Wrote galaxy to " << filename << endl;
}

void loadParticles(const char *filename)
{
	ifstream in;
	in.open(filename);
	if(!in.good()) {
		cout << "Cannot read " << filename << endl;
		return;
	}

	// 1st line:
	// <n> <h> <e2>
	int n;
	in >> n;
	in >> h;
	in >> e2;

    // Rest of the lines:
    // <mass> <position> <velocity> <color> <radius>
    while(in.good()) {
	double mass, posx, posy, posz, velx, vely, velz,
	color1, color2, color3, rad;
	in >> mass;
	in >> posx;
	in >> posy;
	in >> posz;
	in >> velx;
	in >> vely;
	in >> velz;
	in >> color1;
	in >> color2;
	in >> color3;
	in >> rad;
	auto p1 = make_shared<Particle>();
	p1->setMass(mass);
	p1->setRadius(rad);
	p1->setPosition(Vector3d(posx, posy, posz));
	p1->setColor(Vector3f(color1, color2, color3));

	p1->setVelocity(Vector3d(velx, vely, velz));
	particles.push_back(p1);

	masses.push_back(mass);
	radius.push_back(rad);



	positionx.push_back(posx);
	positiony.push_back(posy);
	positionz.push_back(posz);

	velocityx.push_back(velx);
	velocityy.push_back(vely);
	velocityz.push_back(velz);

	colorx.push_back(color1);
	colory.push_back(color2);
	colorz.push_back(color3);


    }
    in.close();
    cout << "Loaded galaxy from " << filename << endl;
}

void createParticles()
{
    srand(0);
    //	t = 0.0;
    //	h = 1e-2;
    e2 = 1e-4;
    h = 1.0;

    auto p = make_shared<Particle>();
    p->setMass(1e-3);
    p->setPosition(Vector3d(0.0, 0.0, 0.0));
    p->setVelocity(Vector3d(0.0, 0.0, 0.0));
    particles.push_back(p);

    auto p2 = make_shared<Particle>();
    double a = 2.0;
    double r = 1.0;
    p2->setMass(1e-6);
    p2->setPosition(Vector3d(1.0, 0.0, 0.0));
    double y = sqrt(1e-3 * (2/r - 1.0/a));
    p2->setVelocity(Vector3d(0.0, y, 0.0));
    particles.push_back(p2);
}

void stepParticles()
{
    //
    // IMPLEMENT ME
    // PreRequisite: particles have been loaded into global variable
    // vector< shared_ptr<Particle> > particles;
    //vector <Vector3d> forces;

//    for (int i = 0; i < particles.size(); i++) {
//        Vector3d force(0.0, 0.0, 0.0);
//        for (int j = 0; j < particles.size(); j++) {
//            shared_ptr<Particle> partI = particles.at(i);
//            shared_ptr<Particle> partJ = particles.at(j);
//            if(j != i) {
//                Eigen::Vector3d rij = partJ->getPosition() - partI->getPosition();
//                double rsquared = pow(rij.norm(), 2);
//                double numerator = partI->getMass() * partJ->getMass();
//                double denominator = pow(rsquared + e2, 3.0/2);
//                force += (numerator * rij)/denominator;
//            }
//        }
//        forces.push_back(force);
//    }

    double *forceX = (double *)malloc(sizeof(double) * positionx.size());
    double *forceY = (double *)malloc(sizeof(double) * positionx.size());
    double *forceZ = (double *)malloc(sizeof(double) * positionx.size());

    double *posX = &positionx[0];
    double *posY = &positiony[0];
    double *posZ = &positionz[0];

    double *mass = &masses[0];

    int sizeX = positionx.size();
    int sizeY = positiony.size();
    int sizeZ = positionz.size();

    // Offload seems to take more time since there is a lot of
    // serial computation in the second for loop

    //#pragma offload target(mic) inout(forceX: length(sizeX)) inout(forceY: length(sizeY)) inout(forceZ: length(sizeZ)) in(posX: length(sizeX)) in(posY: length(sizeY)) in(posZ: length(sizeZ)) in(mass: length(sizeX))
    #pragma omp parallel for
    for (int i = 0; i < sizeX; i++ ) {
	double forcex = 0, forcey = 0, forcez = 0;
	#pragma simd
	for(int j = 0; j < sizeX; j++ ) {
	    if ( j != i ) {
		double rijx = posX[j] - posX[i];
		double rijy = posY[j] - posY[i];
		double rijz = posZ[j] - posZ[i];
		double normalize = pow(rijx*rijx + rijy*rijy + rijz*rijz, 0.5);
		double rsquared = normalize * normalize;
		double numerator = mass[i] * mass[j];
		double denominator = pow(rsquared + e2, 3.0/2);
		double multiplier = numerator/denominator;
		forcex += multiplier * rijx;
		forcey += multiplier * rijy;
		forcez += multiplier * rijz;
	    }
	}
	//forceX.push_back(forcex);
	//forceY.push_back(forcey);
	//forceZ.push_back(forcez);
	forceX[i] = forcex;
	forceY[i] = forcey;
	forceZ[i] = forcez;
    }

   // for(int i = 0; i < particles.size(); i++) {
   //     particles.at(i)->updateParticleVelocity(forces.at(i), h);
   //     particles.at(i)->updateParticlePosition(forces.at(i), h);
   // }

    double *velx = &velocityx[0];
    double *vely = &velocityy[0];
    double *velz = &velocityz[0];

#pragma omp parallel for
    for (int i = 0; i < positionx.size(); i++) {
       velx[i] += (h * 1.0/mass[i]) * forceX[i];
       vely[i] += (h * 1.0/mass[i]) * forceY[i];
       velz[i] += (h * 1.0/mass[i]) * forceZ[i];

       posX[i] += (h * velx[i]);
       posY[i] += (h * vely[i]);
       posZ[i] += (h * velz[i]);
    }

   t += h;

}

int main(int argc, char **argv)
{
	if(argc != 2 && argc != 3) {
		// Wrong number of arguments
		cout << "Usage: Lab09 <RESOURCE_DIR> <(OPTIONAL) INPUT FILE>" << endl;
		cout << "   or: Lab09 <#steps>       <(OPTIONAL) INPUT FILE>" << endl;
		exit(0);
	}
	// Create the particles...
	if(argc == 2) {
		// ... without input file
		createParticles();
	} else {
		// ... with input file
		loadParticles(argv[2]);
	}
	// Try parsing `steps`
	int steps;
	if(sscanf(argv[1], "%i", &steps)) {
		// Success!
	t = h * steps;
		cout << "Running without OpenGL for " << steps << " steps" << endl;
		// Run without OpenGL
		for(int k = 0; k < steps; ++k) {
			stepParticles();
		}
	} else {
		// `steps` could not be parsed
		cout << "Running with OpenGL" << endl;
		// Run with OpenGL until the window is closed
		RESOURCE_DIR = argv[1] + string("/");
		// Set error callback.
		glfwSetErrorCallback(error_callback);
		// Initialize the library.
		if(!glfwInit()) {
			return -1;
		}
		// Create a windowed mode window and its OpenGL context.
		window = glfwCreateWindow(640, 480, "YOUR NAME", NULL, NULL);
		if(!window) {
			glfwTerminate();
			return -1;
		}
		// Make the window's context current.
		glfwMakeContextCurrent(window);
		// Initialize GLEW.
		glewExperimental = true;
		if(glewInit() != GLEW_OK) {
			cerr << "Failed to initialize GLEW" << endl;
			return -1;
		}
		glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
		cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
		cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
		// Set vsync.
		glfwSwapInterval(1);
		// Set keyboard callback.
		glfwSetKeyCallback(window, key_callback);
		// Set char callback.
		glfwSetCharCallback(window, char_callback);
		// Set cursor position callback.
		glfwSetCursorPosCallback(window, cursor_position_callback);
		// Set mouse button callback.
		glfwSetMouseButtonCallback(window, mouse_button_callback);
		// Initialize scene.
		initGL();
		// Loop until the user closes the window.
		while(!glfwWindowShouldClose(window)) {
			// Step simulation.
			stepParticles();
			// Render scene.
			renderGL();
			// Swap front and back buffers.
			glfwSwapBuffers(window);
			// Poll for and process events.
			glfwPollEvents();
		}
		// Quit program.
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	cout << "Elapsed time: " << (t*3.261539827498732e6) << " years" << endl;
	return 0;
}
