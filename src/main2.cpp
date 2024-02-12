// ----------------------------------------------------------------------------
// main2.cpp
//
//  Created on:  Jan 2024
//      Author: Adama Koita
//        Mail: adama.koita@telecom-paris.fr
//
// Description: SPH and ViscoElastic fluids simulator  
//
// Copyright 2021-2024 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES

#define PARALLEL_CPU_VERSION // faster version than FASTVERSION
#define ADAPTATIVE_TIME

#define VISCOELASTIC
//#define SAVEIMAGES

#define IMGUI
#define THREE_D

#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> // Nécessaire pour glm::value_ptr

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.141592
#endif

#define NUM_BOUNDARY_LAYER 0
#ifdef PARTICLES_AS_BOUNDARIES
#ifndef ADAPTATIVE_TIME
#define NUM_BOUNDARY_LAYER 2
#endif

#ifdef ADAPTATIVE_TIME
#define NUM_BOUNDARY_LAYER 2
#endif
#endif


#include "vector.hpp"
#include "vector3.hpp"
#include "Camera.h"


// window parameters
GLFWwindow* gWindow = nullptr;
GLFWwindow* imgui_window = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;



// pointer to the current camera model
std::shared_ptr<Camera> g_cam;

// camera control variables
float g_meshScale = 1.0; // to update based on the mesh size, so that navigation runs at scale
bool g_rotatingP = false;
bool g_panningP = false;
bool g_zoomingP = false;
double g_baseX = 0.0, g_baseY = 0.0;
glm::vec3 g_baseTrans(0.0);
glm::vec3 g_baseRot(0.0);

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = false;
bool gShowVel = false;
bool gApplyGravity = true;
bool gApplyVisco = false;
bool gApplySprings = false;
bool gSPHfluid = true;
bool gDoubleDensity = false;
bool gAdaptativeTime = true;
bool gAddParticleMode = false;
bool gLeftMouseButtonPressed = false;
bool gTemplateblock = true;
int gSavedCnt = 0;

const int kViewScale = 30;

// SPH Kernel function: cubic spline
class CubicSpline {
public:
	explicit CubicSpline(const Real& h = 1.0f)
#ifndef THREE_D
		: _dim(2)
#else
		: _dim(3)
#endif

	{
		setSmoothingLen(h);
	}
	void setSmoothingLen(const Real& h)
	{
		const Real h2 = square(h), h3 = h2 * h;
		_h = h;
		_sr = 2e0 * h;
		_c[0] = 2e0 / (3e0 * h);
		_c[1] = 10e0 / (7e0 * M_PI * h2);
		_c[2] = 1e0 / (M_PI * h3);//1e0 / (pow(M_PI ,1.5)* h3);//
		_gc[0] = _c[0] / h;
		_gc[1] = _c[1] / h;
		_gc[2] = _c[2] / h;
	}
	Real smoothingLen() const { return _h; }
	Real supportRadius() const { return _sr; }

	Real f(const Real l) const
	{
		const Real q = l / _h;
		if (q < 1e0) return _c[_dim - 1] * (1e0 - 1.5 * square(q) + 0.75 * cube(q));
		else if (q < 2e0) return _c[_dim - 1] * (0.25 * cube(2e0 - q));
		return 0;
	}
	Real derivative_f(const Real l) const
	{
		const Real q = l / _h;
		if (q <= 1e0) return _gc[_dim - 1] * (-3e0 * q + 2.25 * square(q));
		else if (q < 2e0) return -_gc[_dim - 1] * 0.75 * square(2e0 - q);
		return 0;
	}

	Real w(const Vec& rij) const { return f(rij.length()); }
	Vec grad_w(const Vec& rij) const { return grad_w(rij, rij.length()); }
	Vec grad_w(const Vec& rij, const Real len) const
	{
		return derivative_f(len) * rij / len;
	}

private:
	unsigned int _dim;
	Real _h, _sr, _c[3], _gc[3];
};

class Particle {
private:
	bool _isBoundary;
	bool _isLeak;
	Vec _position, _velocity;
	Real _pressure, _density;
	//for visco fluid
	Vec _posPrevious;
	Real _pressureVisco, _pressureNear;
	Real _densityVisco, _densityNear;
	std::vector<tIndex> _neighbors;
	std::vector<float> color;
	std::vector<float> velocityLine;
	std::vector < Real> _L;


public:
	Particle(Vec position, bool boundary) : _position(position), _isBoundary(boundary)
	{
		/*velocityLine.reserve(4);
		color.reserve(3);*/
		_isLeak = false;
		_pressure = 0.f;
		_density = .0f;
		_pressureNear = 0.f;
		_densityVisco = 0.f;
		_densityNear = 0.f;
#ifdef THREE_D
		_velocity = Vec(.0f, .0f, .0f);
#else
		_velocity = Vec(.0f, .0f);
#endif

	}
	Particle(Vec position) : _position(position)
	{
		_isBoundary = false;
		/*velocityLine.reserve(4);
		color.reserve(3);*/
		_isLeak = false;
		_pressure = 0.f;
		_density = .0f;
		_pressureNear = 0.f;
		_densityVisco = 0.f;
		_densityNear = 0.f;
#ifdef THREE_D
		_velocity = Vec(.0f, .0f, .0f);
#else
		_velocity = Vec(.0f, .0f);
#endif

	}

	Vec getPosition() const { return _position; }
	void setPosition(const Vec& position) { _position = position; }
	void addToPosition(const Vec& position) { _position += position; }

	bool isBoundary() const { return _isBoundary; }
	void setIsBoundary(bool boundary) { _isBoundary = boundary; }

	bool getIsLeak() const { return _isLeak; }
	void setIsLeak(bool leak) { _isLeak = leak; }

	Vec getVelocity() const { return _velocity; }
	void setVelocity(const Vec& velocity) { _velocity = velocity; }
	void addToVelocity(const Vec& velocity) { _velocity += velocity; }

	Real getPressure() const { return _pressure; }
	void setPressure(const Real pressure) { _pressure = pressure; }
	void addToPressure(const Real pressure) { _pressure += pressure; }

	Real getPressureVisco() const { return _pressureVisco; }
	void setPressureVisco(const Real pressure) { _pressureVisco = pressure; }
	void addToPressureVisco(const Real pressure) { _pressureVisco += pressure; }

	Real getDensity() const { return _density; }
	void setDensity(const Real density) { _density = density; }
	void addToDensity(const Real density) { _density += density; }

	Real getDensityVisco() const { return _densityVisco; }
	void setDensityVisco(const Real density) { _densityVisco = density; }
	void addTODensityVisco(const Real density) { _densityVisco += density; }

	Vec getPositionPrevious() const { return _posPrevious; }
	void setPositionPrevious(const Vec& posPrevious) { _posPrevious = posPrevious; }
	void addToPositionPrevious(const Vec& posPrevious) { _posPrevious += posPrevious; }

	Real getPNear() const { return _pressureNear; }
	void setPNear(const Real pNear) { _pressureNear = pNear; }
	void addToPNear(const Real pNear) { _pressureNear += pNear; }

	Real getDNear() const { return _densityNear; }
	void setDNear(const Real dNear) { _densityNear = dNear; }
	void addToDNear(const Real dNear) { _densityNear += dNear; }

	std::vector<tIndex> getNeighbors() const { return _neighbors; }
	void setNeighbors(const std::vector<tIndex>& newNeighbors) { _neighbors = newNeighbors; }
	void clearNeighbors() { _neighbors.clear(); }
	void addNeighbor(tIndex i) { _neighbors.push_back(i); }
	void changeNeighbors(std::vector<tIndex> n) {
		clearNeighbors();
		_neighbors = n;
	}

	std::vector<float> getColor() const { return color; }
	void setColor(int p, float newColor) { color[p] = newColor; }
	void addToColor(int p, float newColor) { color[p] += newColor; }

	std::vector<float> getVelocityLine() const { return velocityLine; }
	void setVelocityLine(int p, float newVelocityLine) { velocityLine[p] = newVelocityLine; }
	void addToVelocityLine(int p, float newVelocityLine) { velocityLine[p] += newVelocityLine; }
	void changePosPrevious() { _posPrevious = _position; }

	void initL(std::vector<Real> lzero)
	{
		_L = lzero;
	}
	void addtoLlist(Real l)
	{
		_L.push_back(l);
	}
	Real getL(int i) { return _L[i]; }
	std::vector<Real>  getL() { return _L; }
	void setL(int i, Real l) { _L[i] = l; }
	void setL(std::vector<Real> l) { _L = l; }
	void additionToL(int i, Real l) { _L[i] += l; }

};

class SphSolver {
public:
	explicit SphSolver(
		const Real nu = 0.1, const Real h = 0.5f, const Real density = 10,
		const Vec g = Vec(0, -9.8), const Real eta = 0.001f, const Real gamma = 7.0,
		const Real sigma = 10.3f, const Real beta = 1.1f, const Real L0 = 2.f, const Real k_spring = 0.0001f, const Real alpha = 0.008f, const Real gammaSpring = 0.004f) :
		//gammaSpring between 0 et 0.2
		_kernel(h), _nu(nu), _h(h), _d0(density),
		_g(g), _eta(eta), _gamma(gamma),
		//visoelastic constant
		_sigma(sigma), _beta(beta), _L0(L0), _k_spring(k_spring), _alpha(alpha), _gammaSpring(gammaSpring)
	{
		_dt = 0.0004f;
		_mParticle = _m0 / particleCount();
		_c = std::fabs(_g.y) / _eta;
		_k = _d0 * _c * _c / _gamma;
		std::cout << " _k = " << _k << std::endl;
		_maxVel = std::fabs(_g.length());

		//viscoelastic constant
#ifdef VISCOELASTIC
#ifndef THREE_D
		_d0ViscoELas = 10.f; // Diminuer _d0 pourrait rendre le fluide plus compressible, permettant à la gravité d'avoir un impact plus significatif. 
		_dt = 0.0004f;
		_kViscoElas = 0.004;//0.4f;//30.f;
		_k_spring = 0.3f;
		//_alpha = 0.1f;
		_h = .5f;
		_hVisco = 10.5f;

		_kNear = _kViscoElas * 50.f;//0.01;//
		_L0 = 1.5f;
		//_alpha = 0.001f;

		_applyGravity = true;
		_applyViscosity = false;
		_applySprings = false;
		_SPHfluid = false;
#else
		_d0ViscoELas = 10.f; // Diminuer _d0 pourrait rendre le fluide plus compressible, permettant à la gravité d'avoir un impact plus significatif.
		_dt = 0;
		_k =0.04;//30.f;//// 0.4f;
		_k = 20.f;
		_k_spring = 0.003f;
		//_alpha = 0.1f;
		_h = 3.f;
		_hVisco = 1.5f;
		_d0 = 5.f;
		_kNear = _k * 2.f;//0.01;//
		_L0 = _h;//81.5f;
		_nu = 0.06;

		//_alpha = 0.001f;
		_applyGravity = true;
		_applyViscosity = false;
		_applySprings = false;
		_SPHfluid = false;
		_isAdaptativeDT = true;
		_isTemplateBlockParticles = true;

		/*_dt = 0.07f;
		_h = .5f;
		_L0 = _h;
		_k = 0.004;//0.4f;//30.f;
		_kNear = 0.01;// _k * 50.f;

		_k_spring = 3.3f;
		//_alpha = 0.1f;
		_h = .5f;
		_hVisco = 10.5f;

		//_L0 = 2.5f;
		//_alpha = 0.001f;
		_applyGravity = false;
		_applyViscosity = false;
		_applySprings = false;
		_SPHfluid = true;*/
#endif

#ifdef VISCO_FLUID
		_dt = 0.0004f;
		_hVisco = 3.f;
		_d0ViscoELas = 0.1f; // Diminuer _d0 pourrait rendre le fluide plus compressible, permettant à la gravité d'avoir un impact plus significatif.
		//_d0 = _d0ViscoELas;
		_kViscoElas = 30.f;
		//_k_spring = 0.1f;
		//_alpha = 0.1f;
		_h = .5f;
		_hVisco = 1.5f;
		_kNear = _kViscoElas * 10.f;
		_L0 = 3.f;
		//_sigma = 3.f;
		//_beta = 1.f;

		//_alpha = 0.001f;
#endif

#endif
		if (_isAdaptativeDT) {
			_dt = (0.004f * _h) / _g.length();
			
		}
		else {
			_dt = 0.004f;
		}
#ifndef THREE_D
		_m0 = _d0 * _h * _h;
#else
		_m0 = 1.0f;//_m0 = _d0 * _h * _h;//
#endif



	}

	// assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
	// the size of f_width, f_height; each cell is sampled with 2x2 particles.
	void initScene(
		const int res_x, const int res_y, const int f_width, const int f_height)
	{
		_particles.clear();
		_pos.clear();
		_resX = res_x;
		_resY = res_y;
		_mParticle = _m0 / particleCount();
		_particleBoundariesNumber = 0;

		boundaryOffset = 0.0f;

		/*	// set wall for boundary
		  _l = 0.5*_h - boundaryOffset;
		  _r = static_cast<Real>(res_x) - 0.5*_h + boundaryOffset;
		  _b = 0.5*_h - boundaryOffset;
		  _t = static_cast<Real>(res_y) - 0.5*_h + boundaryOffset;*/

		  // set wall for boundary
		_l = 0.25 - boundaryOffset;
		_r = static_cast<Real>(res_x) - 0.25 + boundaryOffset;
		_b = 0.25 - boundaryOffset;
		_t = static_cast<Real>(res_y) - 0.25 + boundaryOffset;

		// sample a 2D fluid mass
		for (int j = 0; j < f_height; ++j) {
			for (int i = 0; i < f_width; ++i) {
				_particles.push_back(Particle(Vec(i + 0.25f + NUM_BOUNDARY_LAYER * _h, j + 0.25f + NUM_BOUNDARY_LAYER * _h)));
				_particles.push_back(Particle(Vec(i + 0.75f + NUM_BOUNDARY_LAYER * _h, j + 0.25f + NUM_BOUNDARY_LAYER * _h)));
				_particles.push_back(Particle(Vec(i + 0.25f + NUM_BOUNDARY_LAYER * _h, j + 0.75f + NUM_BOUNDARY_LAYER * _h)));
				_particles.push_back(Particle(Vec(i + 0.75f + NUM_BOUNDARY_LAYER * _h, j + 0.75f + NUM_BOUNDARY_LAYER * _h)));
			}
		}

		Real p = 0.5f / 2.f;



		// make sure for the other particle quantities
		_pos = std::vector<Vec>(_particles.size());
		_col = std::vector<float>(_particles.size() * 4, 1.0); // RGBA
		_vln = std::vector<float>(_particles.size() * 4, 0.0); // GL_LINES
		leakedParticles.clear();
		std::vector<Real> lzero = std::vector<Real>(_particles.size(), 0);
		for (int i = 0; i < particleCount(); i++) {
			_pos[i] = _particles[i].getPosition();
			_particles[i].initL(lzero);
		}


		updateColor();
		std::cout << "Total Number of particle : " << particleCount() << " | Number of non boundary particles : " << particleCount() - _particleBoundariesNumber << " |  Number of boundary particles : " << _particleBoundariesNumber << std::endl;

	}
#ifdef THREE_D
	///for 3D
	void initScene(const int res_x, const int res_y, const int res_z, const int f_width, const int f_height, const int f_depth) {
		_particles.clear();
		_pos.clear();
		_resX = res_x;
		_resY = res_y;
		_resZ = res_z; // Ajout de la résolution pour la dimension z
		_fWidth = f_width;
		_fHeight = f_height;
		_fDepth = f_depth;
		_particleBoundariesNumber = 0;

		boundaryOffset = 0.25f;

		// Définir les murs pour les limites
		_l = 0.25f - boundaryOffset;
		_r = static_cast<Real>(res_x) - 0.25f + boundaryOffset;
		_b = 0.25f - boundaryOffset;
		_t = static_cast<Real>(res_y) - 0.25f + boundaryOffset;
		_back = 0.25f - boundaryOffset; // Nouvelle limite pour la face avant
		_front = static_cast<Real>(res_z) - 0.25f + boundaryOffset; // Nouvelle limite pour la face arrière

		if (!_isTemplateBlockParticles) {

			float Offset = 0.0f;//0.f;//
			float particleSpace = .5f;
			int bringingTogetherFactor = 1; // make the particle more or less close together without changing their numbers. Different from 0
			//TODO : decommenter si je veux mon set de particules 
			for (float k = 0.; k < f_depth; k += 2 * particleSpace) {
				for (float j = 0; j < f_height; j += 2 * particleSpace) {
					for (float i = 0; i < f_width; i += 2 * particleSpace) {
						// Pour chaque position (i, j), créer plusieurs particules à différentes hauteurs (k)
						for (float zOffset = 0.25; zOffset <= 0.75; zOffset += particleSpace) {

							_particles.push_back(Particle(Vec(i + 0.25f, j + 0.25f, k + zOffset)));
							_particles.push_back(Particle(Vec(i + 0.75f, j + 0.25f, k + zOffset)));
							_particles.push_back(Particle(Vec(i + 0.25f, j + 0.75f, k + zOffset)));
							_particles.push_back(Particle(Vec(i + 0.75f, j + 0.75f, k + zOffset)));

						}
					}
				}
			}
		}else{

		_particles.push_back(Particle(Vec( 0.25f,  0.25f,  0.25f)));
		_particles.push_back(Particle(Vec( 0.75f,  0.25f, 0.25f)));
		_particles.push_back(Particle(Vec( 0.25f,  0.75f, 0.25f)));
		_particles.push_back(Particle(Vec( 0.75f,  0.75f, 0.25f)));
		}

		_pos = std::vector<Vec>(_particles.size());
		_col = std::vector<float>(_particles.size() * 4, 1.0); // RGBA
		_vln = std::vector<float>(_particles.size() * 6, 0.0); // GL_LINES
		std::vector<Real> lzero = std::vector<Real>(_particles.size(), 0);
		leakedParticles.clear();

		for (int i = 0; i < particleCount(); i++) {
			_pos[i] = _particles[i].getPosition();
			_particles[i].initL(lzero);
		}

		updateColor();
		std::cout << "Total Number of particle : " << particleCount() << " | Number of non boundary particles : " << particleCount() - _particleBoundariesNumber << " |  Number of boundary particles : " << _particleBoundariesNumber << std::endl;
	}

#endif
	

	int n = 0;
	void update()
	{

		std::cout << '.' << std::flush;


#ifdef VISCOELASTIC

		buildCellsNeighborhoud_parallel();
		computeAllNeighbors_parallel();
		if (_SPHfluid) {
			computePressureDensity_parallel();
			ApplyForcesAndComputePosition_parallel();
		}

		computeVelocities_viscoelsatic(); //Aply gravity and viscosity
		updatePosition_viscoelastic(); //update previous and predicted position
		if (_applySprings) {
			adjustSprings();
			applySpringDisplacements();
		}

		if (_doubleDensity)
			computePressureDoubleDensity();


		//here resoolve collision
		resolveCollisionBoundary();
		//last
		updateNextVelocityAndColors_viscoelastic();
		updatePosList(); //for the rendering

		if (_isAdaptativeDT) {
			_maxVel += (_g.length() * _dt); //avoird maxVel = 0;

			_dt = (0.4f * _h) / _maxVel; //CFL condition
			//std::cout << "Vmax=" << _maxVel << " dt=" << _dt << std::flush;

			_maxVel = 0.f;
			n++;
		}
#endif


	}
































	Real getD0() { return _d0; }
	Real getK() { return _k; }
	Real getKNearViscoElas() { return _kNear; }
	Real getKSpring() { return _k_spring; }
	Real getH() { return _h; }
	Real getL0() { return _L0; }
	Real getAlpha() { return _alpha; }
	Real getBeta() { return _beta; }
	Real getDt() { return _dt; }
	Real getSigma() { return _sigma; }
	Real getGammaSpring() { return _gammaSpring; }
	Real getNu() { return _nu; }
	int const getLeaksNumber() const { return leakedParticles.size(); }


	void applyGravity(bool q) { _applyGravity = q; }
	bool isGravityApplied() { return _applyGravity; }
	void applyViscosity(bool q) { _applyViscosity = q; }
	bool isViscosityApplied() { return _applyViscosity; }
	void applySprings(bool q) { _applySprings = q; }
	bool isSpringsApplied() { return _applySprings; }
	void applySPH(bool q)
	{
		_SPHfluid = q;

	}
	bool isSPHApplied() { return _SPHfluid; }
	bool isDoubleDensityApplied() { return _doubleDensity; }
	void applyDoubleDensity(bool q)
	{
		_doubleDensity = q;
	
	}
	bool isAdaptativeTime() { return _isAdaptativeDT; }
	void applyAdaptativeTime(bool q)
	{
		if(q == true){
			if(_isAdaptativeDT == false){
				_isAdaptativeDT = true;
				initScene(resX(), resY(), resZ(), width(), height(), depth());}
		}else{
			if(_isAdaptativeDT == true){
				_isAdaptativeDT = false;
				initScene(resX(), resY(), resZ(), width(), height(), depth());}

		}

	}
	bool isTemplateModeParticle() { return _isTemplateBlockParticles; }
	void applyTemplateBlock(bool q)
	{
		
		if (q == true) {
			if (_isTemplateBlockParticles == false){
				_isTemplateBlockParticles = true;
				initScene(resX(), resY(), resZ(), width(), height(), depth());}
		}
		else {
			if (_isTemplateBlockParticles == true){
				_isTemplateBlockParticles = false;
				initScene(resX(), resY(), resZ(), width(), height(), depth());}

		}

	}
	
	void addParticles(const Vec& pos){
		Vec newPos = Vec(
			clamp(pos.x+0.25f, _l + 0.25f, _r - 0.25f),
			clamp(pos.y + 0.25f, _b + 0.25f, _t - 0.25f),
			clamp(pos.z + 0.25f, _back + 0.25f, _front - 0.25f));
		for (Particle& particule : _particles) {
			particule.addtoLlist(0);
		}
		Particle particle = Particle(newPos);
		std::vector<Real> lzero = std::vector<Real>(_particles.size()+1, 0);
		particle.initL(lzero);
		_particles.push_back(particle);

		//update others listes
		//_pos = std::vector<Vec>(_particles.size());
		_pos.push_back(newPos) ;
		_col = std::vector<float>(_particles.size() * 4, 1.0); // RGBA
		_vln = std::vector<float>(_particles.size() * 6, 0.0); // GL_LINES
		updateColor();


	}
	void updateFactors(Real dt, Real d0, Real k, Real kNear, Real kSpring, Real h, Real L0, Real alpha, Real beta, Real sigma, Real gammaSpring, Real nu)
	{
		_dt = dt;
		_d0 = d0;
		_k = k;
		_kNear = kNear;//0.01;//
		_k_spring = kSpring;
		_h = h;
		_L0 = L0;
		_alpha = alpha;
		_beta = beta;
		_sigma = sigma;
		_gammaSpring = gammaSpring;
		_nu = nu;
	}
	//
	tIndex particleCount() const { return _particles.size() - getLeaksNumber(); }
	const Vec& position(const tIndex i) const {
		return _pos[i];
	}
	const float& color(const tIndex i) const { return _col[i]; }
	const float& vline(const tIndex i) const { return _vln[i]; }

	int resX() const { return _resX; }
	int resY() const { return _resY; }
	int resZ() const { return _resZ; }
	int width() const { return _fWidth; }
	int height() const { return _fHeight; }
	int depth() const { return _fDepth; }
	void updatePosList(){
		for (int i = 0; i < particleCount(); i++)
			_pos[i] = _particles[i].getPosition();
	}

	Real getLoopNum()
	{
		if (_isAdaptativeDT){
			if (isFirstStep) {
				isFirstStep = false;
				return 10;
			}

		std::cout << " loop num = " << static_cast<int> (0.01 / _dt) << std::endl;
	}
		return static_cast<int> (0.1 / _dt);

	}

private:

	

	void computeVelocities_viscoelsatic()
	{
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (int i = 0; i < particleCount(); ++i) {
#ifdef PARALLEL_CPU_VERSION
#pragma omp critical
#endif
			{
			if (_applyGravity && !_SPHfluid)
				_particles[i].addToVelocity(_g * _dt);
			if (_applyViscosity)
				applyViscosity_viscoelastic(i); 
		}
		}
	}


	void applyViscosity_viscoelastic(const int& i)
	{

		for (const tIndex& j : _particles[i].getNeighbors()) {
			if (i < j)
			{
				Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();
				Real q = r_ij.length() / _h;
				if (q < 1)
				{
					Real u = (_particles[i].getVelocity() - _particles[j].getVelocity()).dotProduct(r_ij.normalize());
					if (u > 0)
					{
						Vec I = _dt * (1 - q) * ((_sigma * u) + (_beta * u * u)) * r_ij.normalize();
						if (!isBoundary(i))
							_particles[i].addToVelocity(-I / 2.f);
						if (!isBoundary(j))
							_particles[j].addToVelocity(I / 2.f);
					}
				}
			}

		}
	}
	void updatePosition_viscoelastic()
	{
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (int i = 0; i < particleCount(); ++i)
		{
#ifdef PARALLEL_CPU_VERSION
#pragma omp critical
#endif

			 {

			_particles[i].changePosPrevious();
			_particles[i].addToPosition(_dt * _particles[i].getVelocity());

			if (checkLeak(i)) {

				bool isAlreadyLeaked = false;
				for (tIndex leak : leakedParticles)
				{
					if (leak == i) {
						isAlreadyLeaked = true;

					}
				}
				if (!isAlreadyLeaked) {
					leakedParticles.push_back(i);
					std::cout << "A leak happened - Number of lost particle  : " << getLeaksNumber() << std::endl;

				}

			}
		}
		}
	}

	void updateNextVelocityAndColors_viscoelastic()
	{
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (int i = 0; i < particleCount(); ++i)
		{
#ifdef PARALLEL_CPU_VERSION
#pragma omp critical
#endif
			{
			Vec position = _particles[i].getPosition();
			Vec velocity = (position - _particles[i].getPositionPrevious()) / _dt;
			//TODO : remove the comment rehere
			if (!_SPHfluid)
				_particles[i].setVelocity(velocity);

				if (velocity.length() > _maxVel)
					_maxVel = _particles[i].getVelocity().length();
#ifndef VISCO_FLUID
			Real dens;

			_col[i * 4 + 0] = 0.6;
			_col[i * 4 + 1] = 0.6;
			_col[i * 4 + 2] = (_particles[i].getDensity()) / _d0;
			//}
			if (gShowVel) {
#ifndef THREE_D
				//update velocity lines
				_vln[i * 4 + 0] = position.x;
				_vln[i * 4 + 1] = position.y;
				_vln[i * 4 + 2] = position.x + velocity.x;
				_vln[i * 4 + 3] = position.y + velocity.y;
#else
				_vln[i * 6 + 0] = position.x;
				_vln[i * 6 + 1] = position.y;
				_vln[i * 6 + 2] = position.z;
				_vln[i * 6 + 3] = position.x + velocity.x; // Point final basé sur la vitesse
				_vln[i * 6 + 4] = position.y + velocity.y;
				_vln[i * 6 + 5] = position.z + velocity.z;
#endif
			}
#endif
			}

		}
	}

	void adjustSprings()
	{
		float q, d;
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for collapse(2) // Fusionner les deux boucles for imbriquées
#endif

		for (int i = 0; i < particleCount(); ++i)
		{
			for (int j = 0; j < particleCount(); ++j)
			{

				if (i < j)
				{
					Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();
					Real r_ij_length = r_ij.normalize().length();// r_ij.length();
					Real q = r_ij_length / _h;
					if (q < 1) {
						if (_particles[i].getL(j) == 0.f)
							_particles[i].setL(j, _h);

						d = _gammaSpring * _particles[i].getL(j);

						if (r_ij_length > _L0 + d)
						{
							_particles[i].additionToL(j, _dt * _alpha * (r_ij_length - _L0 - d));//std::max(r_ij_length - _L0 - d,0.f);
						}
						else if (r_ij_length < _L0 - d)
						{
							_particles[i].additionToL(j, -_dt * _alpha * (_L0 - d - r_ij_length));//std::max(_L0 - d - r_ij_length,0.f);
						}
					}
				}

				if (_particles[i].getL(j) > _h)
					_particles[i].setL(j, 0.f);
					;
			}
		}
	}

	void applySpringDisplacements()
	{
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for collapse(2) // Fusionner les deux boucles for imbriquées
#endif

		for (int i = 0; i < particleCount(); ++i)
		{
			for (int j = 0; j < particleCount(); ++j)
			{

				if (i < j) {
					Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();
					Vec D = _dt * _dt * _k_spring * (1 - (_particles[i].getL(j) / _h)) * (_particles[i].getL(j) - r_ij.length()) * r_ij.normalize();
					_particles[i].addToPosition(-D / 2);
					_particles[j].addToPosition(D / 2);
				}
			}
		}

	}

	void computePressureDoubleDensity() {
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (tIndex i = 0; i < particleCount(); ++i) {
#ifdef PARALLEL_CPU_VERSION
			#pragma omp critical
#endif

			{
				Vec r_ij; 

				Real density = 0;
				Real densityNear = 0;
				Real q;
				/*Vec r_ij = _particles[i].getPosition() - _particles[i].getPosition();  // Auto-influence
								Real q = r_ij.length() / _h;
				
				if (q < 1) {
					density += (1 - q) * (1 - q);
					densityNear += (1 - q) * (1 - q) * (1 - q);
				}*/
				//for (const tIndex& j : getNeighbors_parallel(i)) {
				std::vector<tIndex> neigh = _particles[i].getNeighbors();
				//std::cout << "particle[" << i << "] neighNumber=" << neigh.size() << std::endl;
				for (const tIndex& j : neigh) {
					r_ij = _particles[i].getPosition() - _particles[j].getPosition();
					q = r_ij.length() / _h;
					//std::cout << "q=" << q << std::endl;
					if (q < 1) {
						density += (1 - q) * (1 - q);
						densityNear += (1 - q) * (1 - q) * (1 - q);
					}
				}

				_particles[i].setDensity(density);
				_particles[i].setDNear(densityNear);

				Real P = _k * (density - _d0);// std::max(_k * (density - _d0), 0.0f);
				Real PNear = _kNear * densityNear;//std::max(_kNear * densityNear, 0.0f);
				_particles[i].setPressure(P);
				_particles[i].setPNear(PNear);
				//std::cout << "k=" << _k << "  knear=" << _kNear << std::endl;

				Vec dx(0.f, 0.f);
				for (const tIndex& j : _particles[i].getNeighbors()) {
					r_ij = _particles[i].getPosition() - _particles[j].getPosition();  // Auto-influence
					Real q = r_ij.length() / _h;
					//std::cout << "q=" << q << std::endl;
					if (q < 1) {
						Vec D = _dt * _dt * ((P * (1 - q)) + (PNear * (1 - q) * (1 - q))) * r_ij.normalize();
						_particles[j].addToPosition(D / 2);
						dx -= D / 2;
					}
				}

				if (!isBoundary(i))
					_particles[i].addToPosition(dx);
			}
		}
	}


	void buildCellsNeighborhoud_parallel() {
		_pidxInGrid.clear();
#ifndef THREE_D
		_pidxInGrid.resize(resX() * resY());
#else
		_pidxInGrid.resize(resX() * resY() * resZ()); // Ajustement pour une grille 3D
#endif

#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for collapse(2) // Fusionner les deux boucles for imbriquées
#endif
		for (int i = 0; i < particleCount(); ++i) {
#ifdef PARALLEL_CPU_VERSION
			#pragma omp critical
#endif
			{
				int cellX = static_cast<int>(_particles[i].getPosition().x);
				int cellY = static_cast<int>(_particles[i].getPosition().y);
#ifdef THREE_D
				int cellZ = static_cast<int>(_particles[i].getPosition().z); // Inclure la coordonnée Z
#endif
#ifndef THREE_D

				if (cellX >= 0 && cellX < _resX && cellY >= 0 && cellY < _resY) {
					_pidxInGrid[idx1d(cellX, cellY)].push_back(i);
#else
				//std::cout << "part " << i << " cellule number: x("<<cellX<<") y("<<cellY<<") z("<<cellZ<<")"<<std::endl;
				if (cellX >= 0 && cellX < _resX && cellY >= 0 && cellY < _resY && cellZ >= 0 && cellZ < _resZ) { // Vérifier les limites pour Z
					_pidxInGrid[idx1d(cellX, cellY, cellZ)].push_back(i); // Utiliser une fonction idx1d adaptée pour 3D
#endif

				}
				}
			}


		}

	std::vector<tIndex> getNeighbors_parallel(tIndex particleIndex) {
		std::vector<tIndex> neighbors;
		const Vec& pos = _particles[particleIndex].getPosition();
		const Real supportRadiusSquared = _kernel.supportRadius() * _kernel.supportRadius();
		int MAX_NEIGHBORS = particleCount(); //1 cells can approximately contain 5 by 5 particle-> Multiple by 9 for a block of neighbour
		neighbors.reserve(MAX_NEIGHBORS);  // Estimation de la taille
		int cellX = static_cast<int>(pos.x);
		int cellY = static_cast<int>(pos.y);
#ifdef THREE_D
		int cellZ = static_cast<int>(pos.z); // Inclure la coordonnée Z
#pragma omp parallel for collapse(3) // Fusionner les trois boucles for imbriquées
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				for (int k = -1; k <= 1; ++k) { // Boucle pour Z
					int neighborCellX = cellX + i;
					int neighborCellY = cellY + j;
					int neighborCellZ = cellZ + k; // Cellule voisine en Z

					if (neighborCellX >= 0 && neighborCellX < _resX && neighborCellY >= 0 && neighborCellY < _resY && neighborCellZ >= 0 && neighborCellZ < _resZ) {
						const std::vector<tIndex>& cell = _pidxInGrid[idx1d(neighborCellX, neighborCellY, neighborCellZ)]; // Utiliser idx1d pour 3D
						for (int n = 0; n < cell.size(); ++n) {
							tIndex neighborIndex = cell[n];
							Vec diff = pos - _particles[neighborIndex].getPosition();
							if (neighborIndex != particleIndex && diff.lengthSquare() < supportRadiusSquared) {
								neighbors.push_back(neighborIndex);
							}
						}
					}
				}
			}
		}
#else
#pragma omp parallel for collapse(2) // Fusionner les deux boucles for imbriquées
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				int neighborCellX = cellX + i;
				int neighborCellY = cellY + j;

				if (neighborCellX >= 0 && neighborCellX < _resX && neighborCellY >= 0 && neighborCellY < _resY) {
					const std::vector<tIndex>& cell = _pidxInGrid[idx1d(neighborCellX, neighborCellY)];
					for (int k = 0; k < cell.size(); ++k) {
						tIndex neighborIndex = cell[k];
						Vec diff = pos - _particles[neighborIndex].getPosition();
						if (neighborIndex != particleIndex && diff.lengthSquare() < supportRadiusSquared) {
							neighbors.push_back(neighborIndex);
						}
					}

				}
			}
		}
#endif
		return neighbors;
	}
	void computeAllNeighbors_parallel() {
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif

		for (tIndex i = 0; i < particleCount(); ++i) {
#ifdef PARALLEL_CPU_VERSION
#pragma omp critical
#endif
			{
				_particles[i].changeNeighbors(getNeighbors(i)); // Mise à jour sécurisée de la structure partagée
				//_neighborsOf.push_back(neighbors);
			}
		}
	}

	void  computeAllNeighborsSimple_viscoelastic() {
		for (tIndex i = 0; i < particleCount(); ++i) {
			std::vector<tIndex> neighbors;
			for (tIndex j = 0; j < particleCount(); j++)
			{
				Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();
				if (std::fabsf(r_ij.length()) < _h)
				{
					neighbors.push_back(j);
				}
			}
			_particles[i].changeNeighbors(neighbors);
		}
	}



	void computePressureDensity_parallel() {
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif

		for (tIndex i = 0; i < particleCount(); ++i) {
			Vec r_ij = _particles[i].getPosition() - _particles[i].getPosition();  // Auto-influence
			Real influence = _kernel.w(r_ij);
			Real density = _m0 * influence;
#ifdef PARALLEL_CPU_VERSION
#pragma omp critical
#endif

			{
			//for (const tIndex& j : getNeighbors_parallel(i)) {
			for (const tIndex& j : _particles[i].getNeighbors()) {

				r_ij = _particles[i].getPosition() - _particles[j].getPosition();
				influence = _kernel.w(r_ij);
				density += _m0 * influence;
			}
			
				_particles[i].setDensity(density);
				::Real pressure = std::max(_k * ((float)pow((density / _d0), _gamma) - 1.0f), 0.0f);
				_particles[i].setPressure(std::max(_k * ((float)pow((density / _d0), _gamma) - 1.0f), 0.0f));
				//std::cout << "particle=" << i << " density=" << density << " pressure="<<pressure<<std::endl;
				
			}
		}
	}

	void ApplyForcesAndComputePosition_parallel() {
		_applyGravity = false;
		std::vector<tIndex> need_res; // for collision
		Vec accel, fpressure, fvisco;
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif

		for (int i = 0; i < particleCount(); i++) {

#ifdef PARALLEL_CPU_VERSION
#pragma omp critical
#endif
				{
					Particle* particle = &(_particles[i]);
					accel = Vec(0);
					fvisco = Vec(0);
					fpressure = Vec(0);

					for (const tIndex& j : particle->getNeighbors()) {
						Vec r_ij = particle->getPosition() - _particles[j].getPosition();
						Vec u_ij = particle->getVelocity() - _particles[j].getVelocity();
						Vec gradW = _kernel.grad_w(r_ij);
						//pressure
						fpressure += gradW * ((particle->getPressure() / (particle->getDensity() * particle->getDensity())) + (_particles[j].getPressure() / (_particles[j].getDensity() * _particles[j].getDensity())));

						//Viscosity
						// avoid to divide by 0
						Real denom = r_ij.dotProduct(r_ij) + (0.01 * _h * _h);
						if (denom != 0.0f)
							fvisco += ((_m0 / _particles[j].getDensity()) * u_ij * (r_ij.dotProduct(gradW) / denom));

					}

				Vec velocity;
				Vec position;
				//std::cout << "pressureF=" << fpressure << " viscosityF" << fvisco << std::endl;
				accel = _g - fpressure + (2.0 * _nu * fvisco);//
				//update velocity
				particle->addToVelocity(_dt * accel);
				velocity = particle->getVelocity();

				particle->addToPosition(_dt * velocity);
				//position = particle->getPosition();

				if (_isAdaptativeDT) {
					//update max velocity (for _dt adapatation )
					if (velocity.length() > _maxVel)
						_maxVel = velocity.length();
				}


#ifndef VISCOELASTIC

					//collision gesture
				if (position.x<_l || position.y<_b || position.x>_r || position.y>_t)
					need_res.push_back(i); {
					for (
						std::vector<tIndex>::const_iterator it = need_res.begin();
						it < need_res.end();
						++it) {
						const Vec p0 = _particles[*it].getPosition();

						_particles[*it].setPosition(Vec(clamp(_particles[*it].getPosition().x, _l, _r),
							clamp(_particles[*it].getPosition().y, _b, _t)));
						_particles[*it].setVelocity((_particles[*it].getPosition() - p0) / _dt);
						//#else

										// Gestion des collisions en 3D
						if (_particles[i].getPosition().x < _l || _particles[i].getPosition().y < _b ||
							_particles[i].getPosition().x > _r || _particles[i].getPosition().y > _t) {
							need_res.push_back(i);
						}

						for (std::vector<tIndex>::iterator it = need_res.begin(); it != need_res.end(); ++it) {
							const Vec p0 = _particles[*it].getPosition();

							// Correction de la position en 3D pour inclure les limites pour Z
							_particles[*it].setPosition(Vec(clamp(_particles[*it].getPosition().x, _l, _r),
								clamp(_particles[*it].getPosition().y, _b, _t)));

							// Mise à jour de la vitesse basée sur la nouvelle position en 3D
							_particles[*it].setVelocity((_particles[*it].getPosition() - p0) / _dt);

						}
					}
				}
#endif

#ifndef VISCOELASTIC

				//update colors
				_col[i * 4 + 0] = 0.6;
				_col[i * 4 + 1] = 0.6;
				_col[i * 4 + 2] = density / _d0;



				//update Velocity lines
				if (gShowVel) {
#ifndef THREE_D
					_vln[i * 4 + 0] = position.x;
					_vln[i * 4 + 1] = position.y;
					_vln[i * 4 + 2] = position.x + velocity.x;
					_vln[i * 4 + 3] = position.y + velocity.y;
#else
					_vln[i * 6 + 0] = position.x;
					_vln[i * 6 + 1] = position.y;
					_vln[i * 6 + 2] = position.z;
					_vln[i * 6 + 3] = position.x + velocity.x; // Point final basé sur la vitesse
					_vln[i * 6 + 4] = position.y + velocity.y;
					_vln[i * 6 + 5] = position.z + velocity.z;
#endif

				}
#endif



			}
		}
	}



	void buildCellsNeighborhoud() {
		// Initialisation : nettoyer les anciennes données
		_pidxInGrid.clear();
		//_pidxInGrid.resize(resX() * resY());
#ifdef THREE_D
		_pidxInGrid.resize(resX() * resY() * resZ()); // Ajusté pour une grille 3D
#endif


		// assign cell to each particle
		for (int i = 0; i < particleCount(); ++i) {
			int cellX = static_cast<int>(_particles[i].getPosition().x);
			int cellY = static_cast<int>(_particles[i].getPosition().y);
#ifdef THREE_D
			int cellZ = static_cast<int>(_particles[i].getPosition().z); // Inclure la coordonnée Z
			if (cellX >= 0 && cellX < _resX && cellY >= 0 && cellY < _resY && cellZ >= 0 && cellZ < _resZ) { // Vérification pour Z
				_pidxInGrid[idx1d(cellX, cellY, cellZ)].push_back(i); // Utilisation de la version 3D de idx1d
#else

			if (cellX >= 0 && cellX < (_resX) && cellY >= 0 && cellY < (_resY)) {
				_pidxInGrid[idx1d(cellX, cellY)].push_back(i);
#endif

			}
			}
		}

	void computeAllNeighbors() {
		for (tIndex i = 0; i < particleCount(); ++i) {
			std::vector<tIndex> neighbors = getNeighbors(i); // Calcul local des voisins
			_particles[i].changeNeighbors(neighbors); // Mise à jour sécurisée de la structure partagée
			//_neighborsOf.push_back(neighbors);
		}
	}
	void computePressureDensity()
	{
		for (tIndex i = 0; i < particleCount(); ++i) {
			//density compute
			Vec r_ij = _particles[i].getPosition() - _particles[i].getPosition();  // Distance entre la particule i et j
			Real influence = _kernel.w(r_ij);
			Real density = _m0 * influence;
			std::vector<tIndex> neigh = getNeighbors(i);
			for (const tIndex& j : neigh) {
				Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();  // Distance entre la particule i et j
				Real influence = _kernel.w(r_ij);

				density += _m0 * influence;
			}
			//density compute
			_particles[i].setDensity(density);

			//pressure compute
			_particles[i].setPressure(std::max(_k * ((float)pow((density / _d0), _gamma) - 1.0f), 0.0f));
		}



	}
	void applyForcesAndComputePosition()
	{

		int thread_num = omp_get_thread_num();
		Vec accel, fpressure, fvisco;
		for (int i = 0; i < particleCount(); i++) {
			accel = Vec(0, 0);
			fpressure = Vec(0, 0);
			fvisco = Vec(0, 0);


				std::vector<tIndex> neigh = getNeighbors(i);
				for (const tIndex& j : neigh) {
					Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();
					Vec u_ij = _particles[i].getVelocity() - _particles[j].getVelocity();
					Vec gradW = _kernel.grad_w(r_ij);
					//pressure
					fpressure += gradW * ((_particles[i].getPressure() / (_particles[i].getDensity() * _particles[i].getDensity())) + (_particles[j].getPressure() / (_particles[j].getDensity() * _particles[j].getDensity())));

					//Viscosity
					// avoid to divide by 0
					Real denom = r_ij.dotProduct(r_ij) + (0.01 * _h * _h);
					if (denom != 0.0f) {
						fvisco += ((_m0 / _particles[j].getDensity())) * u_ij * (r_ij.dotProduct(gradW) / denom);
					}
				}


			accel += _g - fpressure + (2.0 * _nu * fvisco);
			//update velocity

			_particles[i].addToVelocity(_dt * accel);
			//update position 
			_particles[i].addToPosition(_dt * _particles[i].getVelocity());







#ifndef THREE_D
			_particles[i].addToPosition(_dt * _particles[i].getVelocity());
			//collision gesture
			if (_particles[i].getPosition().x<_l || _particles[i].getPosition().y<_b || _particles[i].getPosition().x>_r || _particles[i].getPosition().y>_t)
				need_res.push_back(i);
			for (
				std::vector<tIndex>::const_iterator it = need_res.begin();
				it < need_res.end();
				++it) {
				const Vec p0 = _particles[*it].getPosition();
				_particles[*it].setPosition(Vec(clamp(_particles[*it].getPosition().x, _l, _r),
					clamp(_particles[*it].getPosition().y, _b, _t)));
				_particles[*it].setVelocity((_particles[*it].getPosition() - p0) / _dt);
			}
#endif


#ifndef VISCOELASTIC

			//update colors
			_col[i * 4 + 0] = 0.6;
			_col[i * 4 + 1] = 0.6;
			_col[i * 4 + 2] = _particles[i].getDensity() / _d0;



			//update Velocity lines
			if (gShowVel) {
				_vln[i * 4 + 0] = _particles[i].getPosition().x;
				_vln[i * 4 + 1] = _particles[i].getPosition().y;
				_vln[i * 4 + 2] = _particles[i].getPosition().x + _particles[i].getVelocity().x;
				_vln[i * 4 + 3] = _particles[i].getPosition().y + _particles[i].getVelocity().y;
			}
#endif

		}

	}



	std::vector<tIndex> getNeighbors(tIndex particleIndex) {
		std::vector<tIndex> neighbors;
		const Vec& pos = _particles[particleIndex].getPosition();
#ifndef THREE_D
		int cellX = static_cast<int>(pos.x);
		int cellY = static_cast<int>(pos.y);
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				int neighborCellX = cellX + i;
				int neighborCellY = cellY + j;

				if (neighborCellX >= 0 && neighborCellX < _resX && neighborCellY >= 0 && neighborCellY < _resY) {
					const std::vector<tIndex>& cell = _pidxInGrid[idx1d(neighborCellX, neighborCellY)];
					for (tIndex neighborIndex : cell) {
						if (neighborIndex != particleIndex && (pos - _particles[neighborIndex].getPosition()).length() < _kernel.supportRadius()) {
							neighbors.push_back(neighborIndex);
							//_neighborsOf[idx1dnei(cellX, cellY)].push_back(neighborIndex);
						}
					}
				}
			}
		}
#else
		int cellX = static_cast<int>(pos.x);
		int cellY = static_cast<int>(pos.y);
		int cellZ = static_cast<int>(pos.z); // Inclure la coordonnée Z

		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				for (int k = -1; k <= 1; ++k) { // Itération pour la dimension Z
					int neighborCellX = cellX + i;
					int neighborCellY = cellY + j;
					int neighborCellZ = cellZ + k; // Cellule voisine en Z

					if (neighborCellX >= 0 && neighborCellX < _resX && neighborCellY >= 0 && neighborCellY < _resY && neighborCellZ >= 0 && neighborCellZ < _resZ) {
						const std::vector<tIndex>& cell = _pidxInGrid[idx1d(neighborCellX, neighborCellY, neighborCellZ)]; // Utiliser idx1d pour 3D
						for (tIndex neighborIndex : cell) {
							if (neighborIndex != particleIndex && (pos - _particles[neighborIndex].getPosition()).length() < _kernel.supportRadius()) {
								neighbors.push_back(neighborIndex);
							}
						}
					}
				}
			}
		}
#endif
		//std::cout << "neighbor size=" << neighbors.size() << std::endl;

		return neighbors;
	}


	bool isBoundary(const tIndex & p)
	{
#ifdef PARTICLES_AS_BOUNDARIES
		return (p >= (particleCount() - _particleBoundariesNumber));
#endif

#ifndef PARTICLES_AS_BOUNDARIES
		return false;
#endif


	}

	bool checkLeak(const tIndex & i)
	{
#ifndef THREE_D
		if (_particles[i].getPosition().x<(_l - 2.0f) || _particles[i].getPosition().y<(_b - 2.0f) || _particles[i].getPosition().x>(_r + 2.0f) || _particles[i].getPosition().y>(_t + 2.0f))
#else
		if (_particles[i].getPosition().x < (_l - 2.0f) || _particles[i].getPosition().y < (_b - 2.0f) || _particles[i].getPosition().z < (_back - 2.0f) || _particles[i].getPosition().x >(_r + 2.0f) || _particles[i].getPosition().y >(_t + 2.0f) || _particles[i].getPosition().z >(_front + 2.0f))
#endif

			return true;
		return false;

	}

	void resolveCollisionBoundary(const tIndex & boundary, const tIndex & j) {

		Vec r_ij = _particles[boundary].getPosition() - _particles[j].getPosition();  // Distance entre la particule boundary et j. r_ij.x = dx, r_ij =dy
		Real distance = r_ij.length();

		if (!checkCollision(distance)) return; //if no collision return

		Real radius = _h / 2;
		// collision direction
		Vec norm = r_ij / distance;

		Vec relativeVelocity = _particles[boundary].getVelocity() - _particles[j].getVelocity();

		// relative velocity in the direction of collision
		Real relativeSpeed = relativeVelocity.dotProduct(norm);

		if (relativeSpeed > 0) return;// if particle go away from each other there is no collision so return


		//   restitution factor ( elasticity )
		Real restitution = 0.15; // factor to adjust


		// impulse of collision
		Real impulsion = -(1.0 + restitution) * relativeSpeed;
		impulsion /= ((1.0 / radius) + (1.0 / radius));

		// apply impulsion

		_particles[j].addToVelocity(norm * (impulsion / radius));


		// Update position to avoid overlap
		Real overlap = 0.5 * (distance - _h); // overlap to apply to avoid inter particle penetration

		_particles[j].addToPosition(norm * overlap);


	}
	bool checkCollision(Real distance)
	{
		return (distance < (_h));
	}





	void computeDensity()
	{
		for (tIndex i = 0; i < particleCount(); ++i) {
			Real density = 0.0;
			for (const tIndex& j : getNeighbors(i)) {
				Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();  // Distance entre la particule i et j
				Real distance = r_ij.length();
				Real influence = _kernel.w(r_ij);
				density += _m0 * influence;

			}
			_particles[i].setDensity(density);
		}
	}

	void computePressure()
	{
		for (int i = 0; i < particleCount(); i++) {
			_particles[i].setPressure(std::max(_k * ((float)pow((_particles[i].getDensity() / _d0), _gamma) - 1.0f), 0.0f));
		}
	}

	void applyBodyForce()
	{
		for (int i = 0; i < particleCount(); ++i) {
			_acc[i] += _g;
		}
	}

	void applyPressureForce()
	{
		computePressure();

		for (int i = 0; i < particleCount(); ++i) {
			Vec f(0.f, 0.f);
			for (const tIndex& j : getNeighbors(i)) {
				Vec r_ij = _particles[i].getPosition() - _particles[j].getPosition();  // Distance entre la particule i et j
				Real distance = r_ij.length();
				if (distance < _kernel.supportRadius()) {
					f += _kernel.grad_w(r_ij) * _m0 * ((_particles[i].getPressure() / (_particles[i].getDensity() * _particles[i].getDensity())) + (_particles[j].getPressure() / (_particles[j].getDensity() * _particles[j].getDensity())));
				}
			}
			_acc[i] += -f / _m0;
		}
	}

	void applyViscousForce()
	{

		for (int i = 0; i < particleCount(); ++i) {
			Vec f(0.0, 0.0);

			for (const tIndex& j : getNeighbors(i)) {
				Vec x_ij = _particles[i].getPosition() - _particles[j].getPosition();
				Vec u_ij = _particles[i].getVelocity() - _particles[j].getVelocity();
				Vec gradW = _kernel.grad_w(x_ij);

				// Éviter la division par zéro et assurer la stabilité numérique
				Real denom = x_ij.dotProduct(x_ij) + (0.01 * _h * _h);
				if (denom > 0.0f) {
					f += ((_m0 / _particles[j].getDensity())) * u_ij * (x_ij.dotProduct(gradW) / denom);
				}
			}

			_acc[i] += 2.0 * _nu * f;
		}
	}


	// simple collision detection/resolution for each particle
	void resolveCollisionBoundary()
	{
		std::vector<tIndex> need_res;
#ifndef THREE_D
		for (tIndex i = 0; i < particleCount(); ++i) {
			if (_particles[i].getPosition().x<_l || _particles[i].getPosition().y<_b || _particles[i].getPosition().x>_r || _particles[i].getPosition().y>_t)
				need_res.push_back(i);
		}

		for (
			std::vector<tIndex>::const_iterator it = need_res.begin();
			it < need_res.end();
			++it) {
			const Vec p0 = _particles[*it].getPosition();

			_particles[*it].setPosition(Vec(clamp(_particles[*it].getPosition().x, _l, _r),
				clamp(_particles[*it].getPosition().y, _b, _t)));

			_particles[*it].setVelocity((_particles[*it].getPosition() - p0) / _dt);
#else
		for (tIndex i = 0; i < particleCount(); ++i) {
			if (_particles[i].getPosition().x < _l || _particles[i].getPosition().y < _b || _particles[i].getPosition().z < _back ||
				_particles[i].getPosition().x > _r || _particles[i].getPosition().y > _t || _particles[i].getPosition().z > _front) {
				need_res.push_back(i);
			}
		}

		for (std::vector<tIndex>::const_iterator it = need_res.begin(); it != need_res.end(); ++it) {
			const Vec p0 =_particles[*it].getPosition();

			// Ajustement de la position de la particule pour éviter les dépassements des limites dans toutes les dimensions
			_particles[*it].setPosition(Vec(
				clamp(_particles[*it].getPosition().x, _l, _r),
				clamp(_particles[*it].getPosition().y, _b, _t),
				clamp(_particles[*it].getPosition().z, _back, _front)
			));

			// Mise à jour de la vitesse de la particule en fonction du nouveau déplacement
			_particles[*it].setVelocity(( _particles[*it].getPosition() - p0) / _dt);
#endif

		}
	}

	void updateColor()
	{
		for (tIndex i = 0; i < particleCount(); ++i) {

			_col[i * 4 + 0] = 0.6;
			_col[i * 4 + 1] = 0.6;
			_col[i * 4 + 2] = _particles[i].getDensity() / _d0;

		}
	}

	void updateVelLine()
	{
		for (tIndex i = 0; i < particleCount(); ++i) {
#ifndef THREE_D
			_vln[i * 4 + 0] = position.x;
			_vln[i * 4 + 1] = position.y;
			_vln[i * 4 + 2] = position.x + velocity.x;
			_vln[i * 4 + 3] = position.y + velocity.y;
#else
			_vln[i * 6 + 0] = _particles[i].getPosition().x;
			_vln[i * 6 + 1] = _particles[i].getPosition().y;
			_vln[i * 6 + 2] = _particles[i].getPosition().z;
			_vln[i * 6 + 3] = _particles[i].getPosition().x + _particles[i].getVelocity().x; // Point final basé sur la vitesse
			_vln[i * 6 + 4] = _particles[i].getPosition().y + _particles[i].getVelocity().y;
			_vln[i * 6 + 5] = _particles[i].getPosition().z + _particles[i].getVelocity().z;
#endif
		}
	}

	inline tIndex idx1d(const int& i, const int& j) { return i + j * resX(); }
	inline tIndex idx1d(const int& i, const int& j, const int& k) { return i + (j * resX()) + (k * resX() * resY()); }
	inline tIndex idx1dnei(const int& i, const int& j) { return i + j * MAX_NEIGHBORS; }

	const CubicSpline _kernel;

	// particle data
	std::vector<Particle> _particles; //list of particle
	std::vector<Vec> _pos;

	std::vector<Vec> _acc;      // acceleration
	std::vector<tIndex> leakedParticles; //lost particles

	std::vector< std::vector<tIndex> > _pidxInGrid; // will help you find neighbor particles

	std::vector<float> _col;    // particle color; just for visualization
	std::vector<float> _vln;    // particle velocity lines; just for visualization

	// simulation
	Real _dt;                     // time step

	int _resX, _resY, _resZ, _fWidth, _fHeight ,_fDepth;             // background grid resolution
	int MAX_NEIGHBORS = 4 * 4 * 9; //1 cells can approximately contain 4 by 4 particle-> Multiple by 9 for a block of neighbour

	tIndex _particleBoundariesNumber;
	// wall
	Real _l, _r, _b, _t, _front, _back;          // wall (boundary)
	Real boundaryOffset;



	// SPH coefficients
	Real _nu;                     // viscosity coefficient
	Real _d0;                     // rest density
	Real _h;                      // particle spacing (i.e., diameter)
	Vec _g;                     // gravity

	Real _m0;                     // rest mass
	Real _mParticle;
	Real _k;                      // EOS coefficient

	Real _eta;
	Real _c;                      // speed of sound
	Real _gamma;                  // EOS power factor

	//For _dt integration
	float _maxVel;
	Real isFirstStep = true;


	//viscoelastic
	Real _kViscoElas;					//same function as _k
	Real _kNear;					//EOS near
	Real _d0ViscoELas;				//_d0 for visco elastic sim
	Real _hVisco;					// h for viscofluid

	Real _sigma;					// viscosity factor ( the high it is the more viscous the fluid would be)
	std::vector< std::vector<Real> > _L;			//spring length value between two fluid particles

	Real _beta;					// quadratic dependance compared with vvelocity. Usefull to avoid particle interpenetration by eliminating high intern speed. SHpuld be non nul
	Real _L0;						// spring rest length
	Real _k_spring;				//spring constant
	Real _alpha;					//plasticity constant
	Real _gammaSpring;
	bool _applyGravity;
	bool _applyViscosity;
	bool _applySprings;
	bool _SPHfluid;
	bool _doubleDensity;

	bool _isAdaptativeDT;
	bool _isTemplateBlockParticles;

	};
	bool ctrlPressed;
#ifndef THREE_D
SphSolver gSolver(0.08, 0.5, 1e3, Vec(0, -9.8), 0.01, 7.0);
#else
SphSolver gSolver(0.08, 0.5, 1000, Vec(0, -0.98, 0), 0.01, 7.0);

#endif



Real _hViscoGUI;					// h for viscofluid

Real _sigmaGUI;					// viscosity factor ( the high it is the more viscous the fluid would be)
Real _betaGUI;					// quadratic dependance compared with vvelocity. Usefull to avoid particle interpenetration by eliminating high intern speed. SHpuld be non nul

Real _L0GUI;						// spring rest length
Real _k_springGUI;				//spring constant
Real _alphaGUI;					//plasticity constant
Real _gammaSpringGUI;
Real _kGUI;
Real _kNearGUI;
Real dtGUI;
Real d0GUI;
Real nuGUI;
Real particleSize = 0.333; 
void printHelp()
{
	std::cout <<
		"> Help:" << std::endl <<
		"    Keyboard commands:" << std::endl <<
		"    * H: print this help" << std::endl <<
		"    * P: toggle simulation" << std::endl <<
		"    * R: reset simulation" << std::endl <<
		"    * G: toggle grid rendering" << std::endl <<
		"    * V: toggle velocity rendering" << std::endl <<
		"    * S: save current frame into a file" << std::endl <<
		"    * Q: quit the program" << std::endl;

}


// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow * window, int width, int height)
{
	gWindowWidth = width;
	gWindowHeight = height;
	g_cam->setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
	glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void initImGUI()
{
	dtGUI = gSolver.getDt();
	d0GUI = gSolver.getD0();
	_kGUI = gSolver.getK();
	_kNearGUI = gSolver.getKNearViscoElas();
	_k_springGUI = gSolver.getKSpring();

	_hViscoGUI = gSolver.getH();
	_L0GUI = gSolver.getL0();
	_alphaGUI = gSolver.getAlpha();
	_betaGUI = gSolver.getBeta();

	_sigmaGUI = gSolver.getSigma();
	nuGUI = gSolver.getNu();
	_gammaSpringGUI = gSolver.getGammaSpring();
	gApplyGravity = gSolver.isGravityApplied();
	gApplyVisco = gSolver.isViscosityApplied();
	gApplySprings = gSolver.isSpringsApplied();
	gSPHfluid = gSolver.isSPHApplied();
	gDoubleDensity = gSolver.isDoubleDensityApplied();
	gTemplateblock = gSolver.isTemplateModeParticle();



	// Initialisation d'ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark(); // ou ImGui::StyleColorsClassic();

	// Initialisation des backends ImGui pour GLFW et OpenGL
	ImGui_ImplGlfw_InitForOpenGL(gWindow, true);
	ImGui_ImplOpenGL3_Init("#version 460"); // Utilisez votre version de GLSL ici

}
void resetSim()
{
#ifndef THREE_D
	gSolver.initScene(48, 32, 32, 16);
#else
	//gSolver.initScene(12, 8, 2, 4, 2, 2);//	gSolver.initScene(24, 16, 8, 16, 8, 8);
	 gSolver.initScene(24, 16, 16, 8, 4, 4);//	gSolver.initScene(24, 16, 8, 16, 8, 8);//
	//gSolver.initScene(8, 4, 4, 8, 4, 4);//	gSolver.initScene(24, 16, 8, 16, 8, 8);

#endif


}
// Executed each time a key is entered.
void keyCallback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS && key == GLFW_KEY_H) {
		printHelp();
	}
	else if (action == GLFW_PRESS && key == GLFW_KEY_S) {
		gSaveFile = !gSaveFile;
	}
	else if (action == GLFW_PRESS && key == GLFW_KEY_G) {
		gShowGrid = !gShowGrid;
	}
	else if (action == GLFW_PRESS && key == GLFW_KEY_V) {
		gShowVel = !gShowVel;
	}
	else if (action == GLFW_PRESS && key == GLFW_KEY_P) {
		gAppTimerStoppedP = !gAppTimerStoppedP;
		if (!gAppTimerStoppedP)
			gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
	}
	else if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
		glfwSetWindowShouldClose(window, true);
	}
	else if (action == GLFW_PRESS && key == GLFW_KEY_R) {
		resetSim();
	}
	if (action == GLFW_PRESS && (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL)) {
		ctrlPressed = true;
	}
	else if (action == GLFW_RELEASE && (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL)) {
		ctrlPressed = false;
	}

}

// Called each time the mouse cursor moves
void cursorPosCallback(GLFWwindow * window, double xpos, double ypos)
{
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) {
		return;
	}
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	const float normalizer = static_cast<float>((width + height) / 2);
	const float dx = static_cast<float>((g_baseX - xpos) / normalizer);
	const float dy = static_cast<float>((ypos - g_baseY) / normalizer);
	if (g_rotatingP) {
		const glm::vec3 dRot(-dy * M_PI, dx * M_PI, 0.0);
		g_cam->setRotation(g_baseRot + dRot);
	}
	else if (g_panningP) {
		g_cam->setPosition(g_baseTrans + g_meshScale * glm::vec3(dx, dy, 0.0));
	}
	else if (g_zoomingP) {
		g_cam->setPosition(g_baseTrans + g_meshScale * glm::vec3(0.0, 0.0, dy));
	}
}


Vec screenToWorld(double xpos, double ypos, int screenWidth, int screenHeight) {
	glm::mat4 projection = g_cam->computeProjectionMatrix();
	glm::mat4 view = g_cam->computeViewMatrix() ;
	glm::mat4 viewProjectionInverse = glm::inverse(projection * view);

	double xNormalized = ((2.0 * xpos) / screenWidth) - 1;
	double yNormalized = 1 - ((2.0 * ypos) / screenHeight); // Y inversé

	glm::vec4 screenPos(xNormalized, yNormalized,1.0, 3.0); // Z=1 pour la profondeur
	glm::vec4 worldPos = viewProjectionInverse * screenPos;
	worldPos /= worldPos.w;
	//std::cout << "Souris: (" << xpos << ", " << ypos << "), Normalisé: (" << xNormalized << ", " << yNormalized << "), Monde: (" << worldPos.x << ", " << worldPos.y << ", " << worldPos.z << ")" << std::endl;

	return Vec(worldPos.x -3.3 , worldPos.y + 1.5, worldPos.z - 10);
}

void mouseButtonCallback(GLFWwindow * window, int button, int action, int mods)
{
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) {
		// Let Imgui manage the mouse
		return;
	}

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS ) {
		if (!g_rotatingP && (( !gAddParticleMode) || (gAddParticleMode && ctrlPressed))) {
			g_rotatingP = true;
			glfwGetCursorPos(window, &g_baseX, &g_baseY);
			g_baseRot = g_cam->getRotation();
		}
		gLeftMouseButtonPressed = true;

		
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE){
		gLeftMouseButtonPressed = false;
		g_rotatingP = false;
	}
	else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		if (!g_panningP) {
			g_panningP = true;
			glfwGetCursorPos(window, &g_baseX, &g_baseY);
			g_baseTrans = g_cam->getPosition();
		}
		
	}
	else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
		g_panningP = false;
	}
	else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {
		if (!g_zoomingP) {
			g_zoomingP = true;
			glfwGetCursorPos(window, &g_baseX, &g_baseY);
			g_baseTrans = g_cam->getPosition();
		}
	}
	else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE) {
		g_zoomingP = false;
	}
}
// Callback pour les événements de défilement de la molette de la souris
void scrollCallback(GLFWwindow * window, double xoffset, double yoffset) {
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) {
		// Ne pas traiter les mouvements de la souris pour la scène 3D si ImGui veut capturer la souris
		return;
	}
	// Ajuster la position de la caméra en fonction du défilement de la molette
	// 'yoffset' indique la quantité de défilement vertical
	const float zoomSensitivity = -0.5f; // Sensibilité du zoom, ajustez selon vos besoins
	glm::vec3 camPos = g_cam->getPosition();
	// Ajuster la position de la caméra en Z en fonction du défilement (vous pouvez ajuster pour un autre axe si nécessaire)
	camPos += glm::vec3(0.0, 0.0, yoffset * zoomSensitivity);
	g_cam->setPosition(camPos);
}
void initGLFW()
{
	// Initialize GLFW, the library responsible for window management
	if (!glfwInit()) {
		std::cerr << "ERROR: Failed to init GLFW" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Before creating the window, set some option flags
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

	// Create the window
#ifdef THREE_D
	gWindowWidth = gSolver.resX() * kViewScale * 6;
	gWindowHeight = gSolver.resY() * kViewScale * 6;
#else
	gWindowWidth = gSolver.resX() * kViewScale;
	gWindowHeight = gSolver.resY() * kViewScale;
#endif

	gWindow = glfwCreateWindow(
		gWindowWidth, gWindowHeight,
		"Viscoelatic fluid simulator", nullptr, nullptr);
	if (!gWindow) {
		std::cerr << "ERROR: Failed to open window" << std::endl;
		glfwTerminate();
		std::exit(EXIT_FAILURE);
	}

	// Load the OpenGL context in the GLFW window
	glfwMakeContextCurrent(gWindow);
	gladLoadGL();
	// not mandatory for all, but MacOS X
	glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

	// Connect the callbacks for interactive control
	glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
	glfwSetKeyCallback(gWindow, keyCallback);
	glfwSetCursorPosCallback(gWindow, cursorPosCallback);
	glfwSetMouseButtonCallback(gWindow, mouseButtonCallback);
	glfwSetScrollCallback(gWindow, scrollCallback);

	std::cout << "Window created: " <<
		gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void exitOnCriticalError(const std::string & message)
{
	std::cerr << "> [Critical error]" << message << std::endl;
	std::cerr << "> [Clearing resources]" << std::endl;
	clear();
	std::cerr << "> [Exit]" << std::endl;
	std::exit(EXIT_FAILURE);
}

void initOpenGL()
{
	// Interroger la version d'OpenGL
	const GLubyte* renderer = glGetString(GL_RENDERER); // Obtient le renderer
	const GLubyte* version = glGetString(GL_VERSION); // Obtient la version d'OpenGL

	std::cout << "Renderer: " << renderer << std::endl;
	std::cout << "Version OpenGL: " << version << std::endl;
	// Load extensions for modern OpenGL
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		exitOnCriticalError("[Failed to initialize OpenGL context]");

	glDisable(GL_CULL_FACE);
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
#ifdef THREE_D
	glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, gSolver.resZ());
#else
	glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);

#endif
}
void intiCamera()
{
	// Init camera
	int width, height;
	glfwGetWindowSize(gWindow, &width, &height);
	g_cam = std::make_shared<Camera>();
	g_cam->setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
}
void init()
{


#ifndef THREE_D
	gSolver.initScene(48, 32, 32, 16);
#else
	//gSolver.initScene(6, 4, 2, 2, 2, 2);//	gSolver.initScene(24, 16, 8, 16, 8, 8);
	//gSolver.initScene(12, 8, 2, 4, 2, 2);//	gSolver.initScene(24, 16, 8, 16, 8, 8);
	gSolver.initScene(24, 16, 16, 8, 4, 4);//gSolver.initScene(24, 16, 8, 16, 8, 8);//
	//gSolver.initScene(8, 4, 4, 8, 4, 4);//	gSolver.initScene(24, 16, 8, 16, 8, 8);



#endif
	initGLFW();                   // Windowing system
	initOpenGL();
#ifdef IMGUI
	initImGUI();
#endif
	intiCamera();
	g_meshScale = 24;
	// Adjust the camera to the mesh
	g_cam->setPosition(glm::vec3(1.5, 1.5, 7.5));
	g_cam->setNear(g_meshScale / 100.f);
	g_cam->setFar(6.0 * g_meshScale);
}

void clear()
{
	g_cam.reset();
	glfwDestroyWindow(gWindow);
	// Nettoyage
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
}

void drawAxes(float length) {
	// Sauvegarder l'état actuel de OpenGL
	glPushAttrib(GL_ENABLE_BIT);

	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glLineWidth(2.0f);

	// Commencer à dessiner des lignes
	glBegin(GL_LINES);

	// Axe X en rouge
	glColor3f(1.0f, 0.0f, 0.0f); // Rouge
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(length, 0.0f, 0.0f);

	// Axe Y en vert
	glColor3f(0.0f, 1.0f, 0.0f); // Vert
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, length, 0.0f);

	// Axe Z en bleu
	glColor3f(0.0f, 0.0f, 1.0f); // Bleu
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, length);

	// Finir de dessiner les lignes
	glEnd();

	// Restaurer l'état précédent de OpenGL
	glPopAttrib();
}

// The main rendering call
void render()
{
	glClearColor(.4f, .4f, .4f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//camera
	// Configuration de la matrice de projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity(); // Réinitialise la matrice de projection
	glm::mat4 projection = g_cam->computeProjectionMatrix();
	// Charge la matrice de projection depuis GLM vers OpenGL
	glLoadMatrixf(glm::value_ptr(projection));

	// Configuration de la matrice de vue
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity(); // Réinitialise la matrice de vue
	glm::mat4 view = g_cam->computeViewMatrix();
	// Applique la matrice de vue
	glMultMatrixf(glm::value_ptr(view));
#ifdef IMGUI

	// Commencer la nouvelle frame ImGui
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
#endif

#ifndef THREE_D
	// grid guides
	if (gShowGrid) {
		glBegin(GL_LINES);
		for (int i = 1; i < gSolver.resX(); ++i) {
			glColor3f(0.3, 0.3, 0.3);
			glVertex2f(static_cast<Real>(i), 0.0);
			glColor3f(0.3, 0.3, 0.3);
			glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
		}
		for (int j = 1; j < gSolver.resY(); ++j) {
			glColor3f(0.3, 0.3, 0.3);
			glVertex2f(0.0, static_cast<Real>(j));
			glColor3f(0.3, 0.3, 0.3);
			glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
		}
		glEnd();
	}
#else
	// Grid guides in 3D
	if (gShowGrid) {
		glBegin(GL_LINES);

		// Lignes parallèles à l'axe X
		for (int j = 0; j <= gSolver.resY(); ++j) {
			for (int k = 0; k <= gSolver.resZ(); ++k) {
				glColor3f(0.3, 0.3, 0.3);
				glVertex3f(0.0, static_cast<Real>(j), static_cast<Real>(k));
				glVertex3f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j), static_cast<Real>(k));
			}
		}

		// Lignes parallèles à l'axe Y
		for (int i = 0; i <= gSolver.resX(); ++i) {
			for (int k = 0; k <= gSolver.resZ(); ++k) {
				glColor3f(0.3, 0.3, 0.3);
				glVertex3f(static_cast<Real>(i), 0.0, static_cast<Real>(k));
				glVertex3f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()), static_cast<Real>(k));
			}
		}

		// Lignes parallèles à l'axe Z
		for (int i = 0; i <= gSolver.resX(); ++i) {
			for (int j = 0; j <= gSolver.resY(); ++j) {
				glColor3f(0.3, 0.3, 0.3);
				glVertex3f(static_cast<Real>(i), static_cast<Real>(j), 0.0);
				glVertex3f(static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(gSolver.resZ()));
			}
		}

		glEnd();
	}

	else {
		glEnable(GL_BLEND); // Active la transparence
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Définit le mode de mélange pour la transparence

		glBegin(GL_LINES);

		// Définit la couleur avec transparence
		glColor4f(0.3, 0.3, 0.3, 0.5); // Le dernier composant est l'alpha

		// Dessine uniquement les bords de la grille
		// Bords parallèles à l'axe X
		int maxX = gSolver.resX(), maxY = gSolver.resY(), maxZ = gSolver.resZ();
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(maxX, 0.0, 0.0);
		glVertex3f(0.0, maxY, 0.0);
		glVertex3f(maxX, maxY, 0.0);
		glVertex3f(0.0, 0.0, maxZ);
		glVertex3f(maxX, 0.0, maxZ);
		glVertex3f(0.0, maxY, maxZ);
		glVertex3f(maxX, maxY, maxZ);

		// Bords parallèles à l'axe Y
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, maxY, 0.0);
		glVertex3f(maxX, 0.0, 0.0);
		glVertex3f(maxX, maxY, 0.0);
		glVertex3f(0.0, 0.0, maxZ);
		glVertex3f(0.0, maxY, maxZ);
		glVertex3f(maxX, 0.0, maxZ);
		glVertex3f(maxX, maxY, maxZ);

		// Bords parallèles à l'axe Z
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, 0.0, maxZ);
		glVertex3f(maxX, 0.0, 0.0);
		glVertex3f(maxX, 0.0, maxZ);
		glVertex3f(0.0, maxY, 0.0);
		glVertex3f(0.0, maxY, maxZ);
		glVertex3f(maxX, maxY, 0.0);
		glVertex3f(maxX, maxY, maxZ);

		glEnd();
		glDisable(GL_BLEND);
	}

#endif
	drawAxes(1.0f);

	// render particles
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glPointSize(particleSize* kViewScale);//glPointSize(0.25f * kViewScale);

	glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
#ifndef THREE_D
	glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
#else
	glVertexPointer(3, GL_FLOAT, 0, &gSolver.position(0));
#endif

	glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
#ifndef THREE_D
	// velocity
	if (gShowVel) {
		glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

		glEnableClientState(GL_VERTEX_ARRAY);

		glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
		glDrawArrays(GL_LINES, 0, gSolver.particleCount() * 2);

		glDisableClientState(GL_VERTEX_ARRAY);

	}
#else
	// Velocity vectors in 3D
	if (gShowVel) {
		glColor4f(0.0f, 0.0f, 0.5f, 0.2f);  // Couleur des lignes de vitesse

		glEnableClientState(GL_VERTEX_ARRAY);

		// Assurez-vous que gSolver.vline renvoie un tableau avec des données 3D pour chaque point de la ligne
		glVertexPointer(3, GL_FLOAT, 0, &gSolver.vline(0));  // Utiliser 3 pour des données 3D
		glDrawArrays(GL_LINES, 0, gSolver.particleCount() * 2);  // Nombre de points est toujours le double du nombre de particules

		glDisableClientState(GL_VERTEX_ARRAY);
	}glDisableClientState(GL_VERTEX_ARRAY);


#endif


#ifdef IMGUI
	ImGui::Begin("Debug ImGui of Adama");
	ImGui::Text("Particle count : %d | _d0 : %f | dt : %f", gSolver.particleCount(), gSolver.getD0(),gSolver.getDt());
	ImGui::Text("Particle leaked : %d ", gSolver.getLeaksNumber());
	

	// Checkbox that appears in the window
	ImGui::Checkbox("Not Template block particles (reset the simulation)", &gTemplateblock);
	ImGui::Checkbox("Add particle with mouse", &gAddParticleMode);
	ImGui::Checkbox("saveFile", &gSaveFile);
	ImGui::Checkbox("show grid", &gShowGrid);
	ImGui::Checkbox("show Velocities", &gShowVel);
	ImGui::Checkbox("Apply gravity", &gApplyGravity);
	ImGui::Checkbox("Apply Viscosity", &gApplyVisco);
	ImGui::Checkbox("Apply Springs", &gApplySprings);
	ImGui::Checkbox("SPH fluid", &gSPHfluid);
	ImGui::Checkbox("Double density", &gDoubleDensity); 
	ImGui::Checkbox("Adaptative time (reset  the simulation)", &gAdaptativeTime);

#ifndef THREE_D
	// Slider that appears in the window
	//ImGui::SliderFloat("Particle size", &size, ??);
	ImGui::SliderFloat("k ", &_kGUI, 0.0001f, 30.0f, "Valeur: %.3f", 0.0001f);
	// ImGui::InputFloat(" ", &_kGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("k near", &_kNearGUI, 0.001f, 100.0f, "Valeur: %.3f", 0.001f);
	//ImGui::InputFloat(" ", &_kNearGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("k spring", &_k_springGUI, .0f, 50.0f, "Valeur: %.3f", 0.01f);
	//ImGui::InputFloat(" ", &_k_springGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("gamma spring", &_gammaSpringGUI, .0f, 50.0f, "Valeur: %.3f", 0.01f);
	//ImGui::InputFloat(" ", &_gammaSpringGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("h", &_hViscoGUI, 0.5f, 200.f, "Valeur: %.3f", 0.1f);
	//ImGui::InputFloat(" ", &_hViscoGUI);

	ImGui::SliderFloat("sigma (viscosity)", &_sigmaGUI, 0.5f, 100.f, "Valeur: %.3f", 0.1f);
	//ImGui::InputFloat(" ", &_sigmaGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("L0", &_L0GUI, 0.5f, 100.f, "Valeur: %.3f", 0.1f);
	//ImGui::InputFloat(" ", &_L0GUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("d0", &d0GUI, 0.f, 2000.f, "Valeur: % .3f", 1);
	//ImGui::InputFloat(" ", &d0GUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("dt", &dtGUI, 0.00004f, 1.0f, "Valeur: %.3f", 0.00001f);
	//ImGui::InputFloat(" ", &dtGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("alpha", &_alphaGUI, 0.0001f, 10.0f, "Valeur: %.3f", 0.001f);
	//ImGui::InputFloat(" ", &_alphaGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("beta", &_betaGUI, 0.0005f, 10.0f, "Valeur: %.3f", 0.001f);
	//ImGui::InputFloat(" ", &_betaGUI, 0.1f, 1.0f, "%.3f");
#else
	// Slider that appears in the window
	//ImGui::SliderFloat("Particle size", &size, ??);
	if (!gDoubleDensity) {
		ImGui::SliderFloat("k ", &_kGUI, 0.000f, 100, "Valeur: %.3f", 0.0001f);
		// ImGui::InputFloat(" ", &_kGUI, 0.1f, 1.0f, "%.3f");

		ImGui::SliderFloat("k near", &_kNearGUI, 0.00f, 200, "Valeur: %.3f", 0.001f);
		//ImGui::InputFloat(" ", &_kNearGUI, 0.1f, 1.0f, "%.3f");
	}else {
		ImGui::SliderFloat("k ", &_kGUI, 0.000f, 1, "Valeur: %.3f", 0.0001f);
		// ImGui::InputFloat(" ", &_kGUI, 0.1f, 1.0f, "%.3f");

		ImGui::SliderFloat("k near", &_kNearGUI, 0.00f, 2, "Valeur: %.3f", 0.001f);
		//ImGui::InputFloat(" ", &_kNearGUI, 0.1f, 1.0f, "%.3f");
	}
	
	ImGui::SliderFloat("k spring", &_k_springGUI, .0f, 1.0f, "Valeur: %.3f", 0.01f);
	//ImGui::InputFloat(" ", &_k_springGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("gamma spring", &_gammaSpringGUI, .0f, 0.2f, "Valeur: %.3f", 0.01f);
	//ImGui::InputFloat(" ", &_gammaSpringGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("alpha (spring plasticity)", &_alphaGUI, 0.0001f, 0.5f, "Valeur: %.3f", 0.001f);
	//ImGui::InputFloat(" ", &_alphaGUI, 0.1f, 1.0f, "%.3f");
	ImGui::SliderFloat("L0 (spring)", &_L0GUI, 0.5f, 100.f, "Valeur: %.3f", 0.1f);
	//ImGui::InputFloat(" ", &_L0GUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("h", &_hViscoGUI, 0.5f, 100.f, "Valeur: %.3f", 0.1f);
	//ImGui::InputFloat(" ", &_hViscoGUI)

	ImGui::SliderFloat("sigma (+viscosity)", &_sigmaGUI, 0.5f, 200.f, "Valeur: %.3f", 0.1f);
	//ImGui::InputFloat(" ", &_sigmaGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("beta(- viscosity)", &_betaGUI, 0.0005f, 200.0f, "Valeur: %.3f", 0.001f);
	//ImGui::InputFloat(" ", &_betaGUI, 0.1f, 1.0f, "%.3f");

	

	ImGui::SliderFloat("d0", &d0GUI, 0.f, 20.f);
	//ImGui::InputFloat(" ", &d0GUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("dt", &dtGUI, 0.0001f, 0.5f, "Valeur: %.3f", 0.0001f);
	//ImGui::InputFloat(" ", &dtGUI, 0.1f, 1.0f, "%.3f");


	ImGui::SliderFloat("nu (SPH)", &nuGUI, 0.001f, 5.0f, "Valeur: %.3f", 0.001f);
	//ImGui::InputFloat(" ", &_alphaGUI, 0.1f, 1.0f, "%.3f");

	ImGui::SliderFloat("particle size view", &particleSize, 0.1f, 1.5f, "Valeur: %.3f", 0.001f);
	//ImGui::InputFloat(" ", &_alphaGUI, 0.1f, 1.0f, "%.3f");
#endif


	// Fancy color editor that appears in the window
	//ImGui::ColorEdit4("Color", color);
	ImGui::End();

	// Rendu d'ImGui
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif

	//draw axis


	if (gSaveFile) {
		std::stringstream fpath;
		fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

		std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
		const short int w = gWindowWidth;
		const short int h = gWindowHeight;
		std::vector<int> buf(w * h * 3, 0);
		glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

		FILE* out = fopen(fpath.str().c_str(), "wb");
		short TGAhead[] = { 0, 2, 0, 0, 0, 0, w, h, 24 };
		fwrite(&TGAhead, sizeof(TGAhead), 1, out);
		fwrite(&(buf[0]), 3 * w * h, 1, out);
		fclose(out);
		gSaveFile = false;

		std::cout << "Done" << std::endl;
	}
}

// Update any accessible variable based on the current time
void update(const float currentTime)
{
#ifdef IMGUI 
	gSolver.updateFactors(dtGUI, d0GUI, _kGUI, _kNearGUI, _k_springGUI,
		_hViscoGUI, _L0GUI, _alphaGUI, _betaGUI, _sigmaGUI, _gammaSpringGUI, nuGUI);
	gSolver.applyGravity(gApplyGravity);
	gSolver.applyViscosity(gApplyVisco);
	gSolver.applySprings(gApplySprings);
	gSolver.applyDoubleDensity(gDoubleDensity);
	gSolver.applySPH(gSPHfluid);
	gSolver.applyTemplateBlock(gTemplateblock);
	gSolver.applyAdaptativeTime(gAdaptativeTime);

#endif
	if (!gAppTimerStoppedP) {
		// NOTE: When you want to use application's dt ...
		/*
		const float dt = currentTime - gAppTimerLastClockTime;
		gAppTimerLastClockTime = currentTime;
		gAppTimer += dt;
		*/

		//save a pic after n step


#ifdef SAVEIMAGES
		int n;
		n = gSolver.getLoopNum();// 50;
		// solve 10 steps for better stability ( chaque step est un pas de temps )
		for (int i = 0; i < n; ++i)
#endif

		

		gSolver.update();


#ifdef SAVEIMAGES

		std::stringstream fpath;
		fpath << "withTimeIntegration2" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

		//std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
		const short int w = gWindowWidth;
		const short int h = gWindowHeight;
		std::vector<int> buf(w * h * 3, 0);
		glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

		FILE* out = fopen(fpath.str().c_str(), "wb");
		short TGAhead[] = { 0, 2, 0, 0, 0, 0, w, h, 24 };
		fwrite(&TGAhead, sizeof(TGAhead), 1, out);
		fwrite(&(buf[0]), 3 * w * h, 1, out);
		fclose(out);
#endif

	}
	if (gLeftMouseButtonPressed && gAddParticleMode && !ctrlPressed ) {
		double xpos, ypos;
		int width, height;
		glfwGetWindowSize(gWindow, &width, &height);

		glfwGetCursorPos(gWindow, &xpos, &ypos);

		const float dy = g_cam->getPosition().z/ 2;
		Vec clickPos = screenToWorld(xpos* dy, ypos* dy, width, height);
		gSolver.addParticles(clickPos);
	}
}

int main(int argc, char** argv)
{
	init();
	while (!glfwWindowShouldClose(gWindow)) {
		update(static_cast<float>(glfwGetTime()));
		render();
		glfwSwapBuffers(gWindow);
		glfwPollEvents();
	}
	clear();
	std::cout << " > Quit" << std::endl;
	return EXIT_SUCCESS;
}