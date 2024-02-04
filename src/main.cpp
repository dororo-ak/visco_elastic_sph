// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
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

//#define  FASTVERSION //faster version by merging some function
#define PARALLEL_CPU_VERSION // faster version than FASTVERSION
//#define ADAPTATIVE_TIME

#define VISCOELASTIC
//#define SAVEIMAGES

//#define  PARTICLES_AS_BOUNDARIES //work only if fast version defined
//#define WITHOUT_GRAVITY
//#define VISCO_FLUID
#define IMGUI

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


#include "Vector.hpp"
typedef Vec2f vec;
// window parameters
GLFWwindow *gWindow = nullptr;
GLFWwindow* imgui_window = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;
int imguiWindowWidth = 400;
int imguiWindowHeight = 300;

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
bool gSPHfluid = false;

int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline {
public:
  explicit CubicSpline(const Real& h =1.0f) : _dim(2)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const Real& h)
  {
    const Real h2 = square(h), h3 = h2*h;
    _h = h;
    _sr = 2e0*h;
    _c[0]  = 2e0/(3e0*h);
    _c[1]  = 10e0/(7e0*M_PI*h2);
    _c[2]  = 1e0/(M_PI*h3);
    _gc[0] = _c[0]/h;
    _gc[1] = _c[1]/h;
    _gc[2] = _c[2]/h;
  }
  Real smoothingLen() const { return _h; }
  Real supportRadius() const { return _sr; }

  Real f(const Real l) const
  {
    const Real q = l/_h;
    if(q<1e0) return _c[_dim-1]*(1e0 - 1.5*square(q) + 0.75*cube(q));
    else if(q<2e0) return _c[_dim-1]*(0.25*cube(2e0-q));
    return 0;
  }
  Real derivative_f(const Real l) const
  {
    const Real q = l/_h;
    if(q<=1e0) return _gc[_dim-1]*(-3e0*q+2.25*square(q));
    else if(q<2e0) return -_gc[_dim-1]*0.75*square(2e0-q);
    return 0;
  }

  Real w(const Vec2f &rij) const { return f(rij.length()); }
  Vec2f grad_w(const Vec2f &rij) const { return grad_w(rij, rij.length()); }
  Vec2f grad_w(const Vec2f &rij, const Real len) const
  {
    return derivative_f(len)*rij/len;
  }

private:
  unsigned int _dim;
  Real _h, _sr, _c[3], _gc[3];
};

class Particle {
private:
	bool _isBoundary;
	bool isLeak;
	Vec2f _position, _velocity;
	Real _pressure, _density;
	//for visco fluid
	Vec2f _posPrevious;
	Real _pNear;
	Real _dNear;
	std::vector<tIndex> neighbors;
	float color;
	float velocityLine;
	

public:
	Particle(Vec2f position, bool boundary) : _position(position), _isBoundary(boundary) {}
	Particle(Vec2f position) : _position(position) {}

	Vec2f getPosition() const { return _position; }
	void setPosition(const Vec2f& position) { _position = position; }

	bool isBoundary() const { return _isBoundary; }
	void setIsBoundary(bool boundary) { _isBoundary = boundary; }

	bool getIsLeak() const { return isLeak; }
	void setIsLeak(bool leak) { isLeak = leak; }

	Vec2f getVelocity() const { return _velocity; }
	void setVelocity(const Vec2f& velocity) { _velocity = velocity; }

	Real getPressure() const { return _pressure; }
	void setPressure(const Real pressure) { _pressure = pressure; }

	Real getDensity() const { return _density; }
	void setDensity(const Real density) { _density = density; }

	Vec2f getPositionPrevious() const { return _posPrevious; }
	void setPositionPrevious(const Vec2f& posPrevious) { _posPrevious = posPrevious; }

	Real getPNear() const { return _pNear; }
	void setPNear(const Real pNear) { _pNear = pNear; }

	Real getDNear() const { return _dNear; }
	void setDNear(const Real dNear) { _dNear = dNear; }

	std::vector<tIndex> getNeighbors() const { return neighbors; }
	void setNeighbors(const std::vector<tIndex>& newNeighbors) { neighbors = newNeighbors; }

	float getColor() const { return color; }
	void setColor(float newColor) { color = newColor; }

	float getVelocityLine() const { return velocityLine; }
	void setVelocityLine(float newVelocityLine) { velocityLine = newVelocityLine; }
};

class SphSolver {
public:
  explicit SphSolver(
    const Real nu=0.01, const Real h=0.5f, const Real density=1e3,
    const Vec2f g=Vec2f(0, -9.8), const Real eta=0.01f, const Real gamma=7.0,
    const Real sigma = 1.3f, const Real beta = 1.1f, const Real L0 = 2.f, const Real k_spring = 0.00001f, const Real alpha = 0.3f, const Real gammaSpring = 0.2f) :
	//gammaSpring between 0 et 0.2
    _kernel(h), _nu(nu),_h(h), _d0(density),
    _g(g), _eta(eta), _gamma(gamma),
	//visoelastic constant
	_sigma(sigma), _beta(beta), _L0(L0), _k_spring(k_spring), _alpha(alpha), _gammaSpring(gammaSpring)
  {
	  _dt = 0.0004f;

  	//viscoelastic constant
#ifdef VISCOELASTIC

	  _d0ViscoELas = 10.f; // Diminuer _d0 pourrait rendre le fluide plus compressible, permettant à la gravité d'avoir un impact plus significatif. 
	  _dt = 0.007f;
	  _kViscoElas = 0.004;//0.4f;//30.f;
	  _k_spring = 0.3f;
	  //_alpha = 0.1f;
	  _h = .5f;
	  _hVisco = 10.5f;
	   
	  _kViscoElasNear = _kViscoElas * 50.f;//0.01;//
	  _L0 = 1.5f;
	  //_alpha = 0.001f;
	  _applyGravity = true;
  	  _applyViscosity=false;
  	  _applySprings = false;
  	  _SPHfluid = false;
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
	  _kViscoElasNear = _kViscoElas * 10.f;
	  _L0 =3.f;
	  //_sigma = 3.f;
	  //_beta = 1.f;

	  //_alpha = 0.001f;
#endif

#endif
#ifdef ADAPTATIVE_TIME
  	_dt = 0.0f;
#endif

    _m0 = _d0*_h*_h;
    _c = std::fabs(_g.y)/_eta;
    _k = _d0*_c*_c/_gamma;
	_maxVel = std::fabs(_g.length());

	

  }

  // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
  // the size of f_width, f_height; each cell is sampled with 2x2 particles.
  void initScene(
    const int res_x, const int res_y, const int f_width, const int f_height)
  {
  	_pos.clear();
    _resX = res_x;
    _resY = res_y;
	
	boundaryOffset =0.0f;

  /*	// set wall for boundary
    _l = 0.5*_h - boundaryOffset;
    _r = static_cast<Real>(res_x) - 0.5*_h + boundaryOffset;
    _b = 0.5*_h - boundaryOffset;
    _t = static_cast<Real>(res_y) - 0.5*_h + boundaryOffset;*/

  	// set wall for boundary
	_l = 0.25  - boundaryOffset;
	_r = static_cast<Real>(res_x) - 0.25  + boundaryOffset;
	_b = 0.25  - boundaryOffset;
	_t = static_cast<Real>(res_y) - 0.25  + boundaryOffset;


    // sample a fluid mass
    for(int j=0; j<f_height; ++j) {
      for(int i=0; i<f_width; ++i) {
        _pos.push_back(Vec2f(i+0.25f+ NUM_BOUNDARY_LAYER *_h, j+0.25f+ NUM_BOUNDARY_LAYER *_h));
        _pos.push_back(Vec2f(i+0.75f+ NUM_BOUNDARY_LAYER *_h, j+0.25f+ NUM_BOUNDARY_LAYER *_h));
        _pos.push_back(Vec2f(i+0.25f+ NUM_BOUNDARY_LAYER *_h, j+0.75f+ NUM_BOUNDARY_LAYER *_h));
        _pos.push_back(Vec2f(i+0.75f+ NUM_BOUNDARY_LAYER *_h, j+0.75f+ NUM_BOUNDARY_LAYER *_h));
      }
    }
	Real p = 0.5f/2.f;

#ifdef PARTICLES_AS_BOUNDARIES
	_particleBoundariesNumber = 0;
	//build the bondaries with particles//fill the horizontal boundaries
	for (int i = _l+1; i < res_x; i++) {
		//bottom line
		if (i < _r - 1) {


			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(i + (p), _b + (0.5f * j)));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(i + (3 * p), _b + (0.5f * j)));
				_particleBoundariesNumber++;
			}
	}
        
		//top line
        if(i<_r-1){

			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(i + (p), (_t - (0.5f * j))));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(i + (3 * p), (_t- (0.5f * j))));
				_particleBoundariesNumber++;
			}

        }
	}
	//build the bondaries with particles//fill the vertical boundaries
    for (int i = _b; i < res_y;i++) {
		//bottom line
        if (i < _t) {

			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(_l + (0.5f * j), i + (p)));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(_l+ (0.5f * j), i + (3 * p)));
				_particleBoundariesNumber++;
			}


        }
		//top line
        if (i < _t) {

			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(_r - (0.5f * j), i + (p)));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(_r - (0.5f * j), i + (3 * p)));
				_particleBoundariesNumber++;
			}

        }

	}
#endif

    // make sure for the other particle quantities
    _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _p   = std::vector<Real>(_pos.size(), 0);
    _d   = std::vector<Real>(_pos.size(), 0);
    _col = std::vector<float>(_pos.size()*4, 1.0); // RGBA
    _vln = std::vector<float>(_pos.size()*4, 0.0); // GL_LINES
	_dNear = std::vector<Real>(_pos.size(), 0);
	_pNear = std::vector<Real>(_pos.size(), 0);
	_posPrevious = _pos;
	std::vector<Real> lzero = std::vector<Real>(_pos.size(), 0);
	_L = std::vector<std::vector<Real>>(_pos.size() , lzero);


    updateColor();
	std::cout<<"Total Number of particle : "<<particleCount() << " | Number of non boundary particles : " << particleCount() - _particleBoundariesNumber << " |  Number of boundary particles : " << _particleBoundariesNumber << std::endl;

  }
  int n = 0;
  void update()
  {

	  std::cout << '.' << std::flush;

#ifndef FASTVERSION
#ifndef VISCOELASTIC
	  buildCellsNeighborhoud();
	  computeDensity();
	  computePressure();
	  _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
	  applyBodyForce();
	  applyPressureForce();
	  applyViscousForce();
	  updateVelocity();
	  updatePosition();
	  resolveCollisionBoundary();

	  updateColor();
	  if (gShowVel) updateVelLine();
#endif
#endif
#ifdef FASTVERSION

#ifndef  PARALLEL_CPU_VERSION
#ifndef VISCOELASTIC
	  buildCellsNeighborhoud();
	  //computeAllNeighbors();
	  computePressureDensity();
	  applyForcesAndComputePosition();
#endif
#endif

#ifdef PARALLEL_CPU_VERSION
#ifndef VISCOELASTIC
	  buildCellsNeighborhoud_parallel();
	  computeAllNeighbors_parallel();
	  computePressureDensity_parallel();
	  parallelApplyForcesAndComputePosition();
#endif
#endif



#ifdef ADAPTATIVE_TIME
#ifndef VISCOELASTIC
	  if (isFirstStep)
	  {
		  _maxVel += _g.length() * 100;
		  isFirstStep = false;
	  }
	  else
	  {
		  _maxVel += _g.length();
	  }
	  _dt = (0.04f * _h) / _maxVel; //CFL condition
	  //std::cout << "Vmax=" << _maxVel <<" dt="<<_dt<< std::flush;
	  _maxVel = 0.f;
	  n++;

#endif
#endif



#endif

#ifdef VISCOELASTIC
#ifdef PARALLEL_CPU_VERSION
	  buildCellsNeighborhoud_parallel();
	  computeAllNeighbors_parallel();
#endif

#ifndef PARALLEL_CPU_VERSION
	  buildCellsNeighborhoud();
	  computeAllNeighbors();
	  //computeAllNeighborsSimple_viscoelastic();
#endif

	  computeVelocities_viscoelsatic();
	  updatePosition_viscoelastic();
	  //heree put adjustspring and applyspringdisplacement Sans le spring adjust les particule se repoussent entre elles
	  if (_applySprings) {
		  adjustSprings();
		  applySpringDisplacements();
	  }


	  computePressureDoubleDensity();


	  //here i add this method beacuse i want my fluid to have more a liquid fluid behavior
	  if (_SPHfluid) {
		  computePressureDensity();
		  applyForcesAndComputePosition();
  }


	//here resoolve collision
	resolveCollisionBoundary();

	//last
	updateNextVelocityAndColors_viscoelastic();


#ifdef  ADAPTATIVE_TIME
#ifndef VISCO_FLUID
	//_maxVel += _g.length() * _dt;

	_dt = (0.4f * _h) / _maxVel; //CFL condition
	//std::cout << "Vmax=" << _maxVel <<" dt="<<_dt<< std::flush;
	_maxVel = 0.f;
	n++;
#endif
#endif
#endif


  }

  Real getD0ViscoElas() { return _d0ViscoELas; }
  Real getKViscoElas() {return _kViscoElas;}
  Real getKNearViscoElas() { return _kViscoElasNear; }
  Real getKSpring() { return _k_spring; }
  Real getHViscoElas() { return _hVisco; }
  Real getL0() { return _L0; }
  Real getAlpha() { return _alpha; }
  Real getBeta() { return _beta; }
  Real getDt() { return _dt; }
  Real getSigma() { return _sigma; }
  Real getGammaSpring() { return _gammaSpring; }

  void applyGravity(bool q) { _applyGravity = q; }
  bool isGravityApplied() { return _applyGravity; }
  void applyViscosity(bool q) { _applyViscosity = q; }
  bool isViscosityApplied() { return _applyViscosity; }
  void applySprings(bool q) { _applySprings = q; }
  bool isSpringsApplied() { return _applySprings; }
  void applySPH(bool q) { _SPHfluid = q; }
  bool isSPHApplied() { return _SPHfluid; }
  void updateFactors(Real dt , Real d0, Real k, Real kNear, Real kSpring, Real h, Real L0, Real alpha, Real beta, Real sigma, Real gammaSpring)
  {	  _dt = dt;
	  _d0ViscoELas = d0;
	  _kViscoElas = k;
	  _kViscoElasNear = kNear;//0.01;//
	  _k_spring = kSpring;
	  _hVisco =h;
	  _L0 = L0;
	  _alpha = alpha;
	  _beta = beta;
	  _sigma = sigma;
	  _gammaSpring = gammaSpring;
	  std::cout << '.' << std::flush;
  }
  //
  tIndex particleCount() const { return _pos.size(); }
  const Vec2f& position(const tIndex i) const { return _pos[i]; }
  const float& color(const tIndex i) const { return _col[i]; }
  const float& vline(const tIndex i) const { return _vln[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }


  Real getLoopNum()
  {
  	#ifdef ADAPTATIVE_TIME
	  if (isFirstStep){
		  isFirstStep = false;
		  return 10;
	  }

	  std::cout << " loop num = " << static_cast<int> (0.01 / _dt) << std::endl;
#endif
	  return static_cast<int> (0.1 / _dt);

  }

private:




	void computeVelocities_viscoelsatic()
	{
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (int i = 0; i < particleCount(); ++i) {
			if (!isBoundary(i))
				if(_applyGravity)
					_vel[i] += _g * _dt;
			if(_applyViscosity)
				applyViscosity_viscoelastic(i); //TODO : put the code here for better performences
		}
	}


	void applyViscosity_viscoelastic(const int& i)
	{

		for (const tIndex& j : _neighborsOf[i]) {
			if(i <j)
			{
				Vec2f r_ij = _pos[i] - _pos[j];
				Real q = r_ij.length() / _hVisco;
				if(q<1)
				{
					Real u = (_vel[i] - _vel[j]).dotProduct(r_ij);
					if(u>0)
					{
						Vec2f I = _dt * (1 - q) * ((_sigma * u) + (_beta * u * u) )* r_ij.normalize();
						if (!isBoundary(i)) 
							_vel[i] -= I / 2.f;
						if (!isBoundary(j)) 
							_vel[j] += I / 2.f;
					}
				}
			}

		}
	}
	void updatePosition_viscoelastic()
	{
		_posPrevious = _pos;
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (int i = 0; i < particleCount(); ++i)
		{
			if (!isBoundary(i)) 
			_pos[i] += _dt * _vel[i];
		}
		
	}

	void updateNextVelocityAndColors_viscoelastic()
	{
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (int i = 0; i < particleCount(); ++i)
		{
			if (!isBoundary(i))
				_vel[i] = (_pos[i] - _posPrevious[i]) / _dt;
			if (_vel[i].length() > _maxVel)
				_maxVel = _vel[i].length();
#ifndef VISCO_FLUID
			//update colors
			_col[i * 4 + 0] = 0.6;
			_col[i * 4 + 1] = 0.6;
			_col[i * 4 + 2] = (_d[i]+ _dNear[i]) / _d0ViscoELas;

			//update velocity lines
			_vln[i * 4 + 0] = _pos[i].x;
			_vln[i * 4 + 1] = _pos[i].y;
			_vln[i * 4 + 2] = _pos[i].x + _vel[i].x;
			_vln[i * 4 + 3] = _pos[i].y + _vel[i].y;
#endif


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
					Vec2f r_ij = _pos[i] - _pos[j];
					Real r_ij_length = r_ij.length();
					Real q = r_ij_length / _hVisco;
					if (q < 1) {
						if (_L[i][j] == 0.f)
							_L[i][j] = _hVisco;
						d = _gammaSpring * _L[i][j];

						if (r_ij_length > _L0 + d)
						{
							_L[i][j] += _dt * _alpha * (r_ij_length - _L0 - d);//std::max(r_ij_length - _L0 - d,0.f);
						}
						else if (r_ij_length < _L0 - d)
						{
							_L[i][j] -= _dt * _alpha * (_L0 - d - r_ij_length);//std::max(_L0 - d - r_ij_length,0.f);
						}
					}
				}
				if (_L[i][j] > _hVisco)
					_L[i][j] = 0.f;
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
					Vec2f r_ij = _pos[i] - _pos[j];
					Vec2f D = _dt * _dt * _k_spring * (1 - (_L[i][j] / _hVisco)) * (_L[i][j] - r_ij.length()) * r_ij.normalize();
					_pos[i] -= D / 2;
					_pos[j] += D / 2;
				}
			}
		}

	}

	void computePressureDoubleDensity() {
#ifdef PARALLEL_CPU_VERSION
#pragma omp parallel for
#endif
		for (tIndex i = 0; i < particleCount(); ++i) {
			Vec2f r_ij = _pos[i] - _pos[i];  // Auto-influence
			//Vec2f r_ij; 
			Real q = r_ij.length() / _hVisco;
			Real density = 0;
			Real densityNear = 0;
			if (q < 1) {
				density += (1 - q) * (1 - q);
				densityNear += (1 - q) * (1 - q) * (1 - q);
			}
			//for (const tIndex& j : getNeighbors_parallel(i)) {
			std::vector<tIndex> neigh = _neighborsOf[i];
			//std::cout << "particle[" << i << "] neighNumber=" << neigh.size() << std::endl;
			for (const tIndex& j : neigh) {
				r_ij = _pos[i] - _pos[j];
				q = r_ij.length() / _hVisco;
				//std::cout << "q=" << q << std::endl;
				if (q < 1) {
					density += (1 - q) * (1 - q);
					densityNear +=  (1 - q) * (1 - q) * (1 - q);
				}
			}

			_d[i] = density;
			_dNear[i] = densityNear;
			//std::cout << "Density=" << density << "  DensityNear=" << densityNear << std::endl;

			Real P = std::max(_kViscoElas * (density - _d0ViscoELas), 0.0f);
			Real PNear = std::max(_kViscoElasNear * densityNear, 0.0f);
			_p[i] = P;
			_pNear[i] = PNear;

			Vec2f dx(0.f, 0.f);
			for (const tIndex& j : _neighborsOf[i]) {
				r_ij = _pos[i] - _pos[j];  // Auto-influence
				Real q = r_ij.length() / _hVisco;
				//std::cout << "q=" << q << std::endl;
				if (q < 1) {
					Vec2f D =  _dt * _dt *((P * (1 - q)) + (PNear * (1 - q) * (1 - q))) * r_ij.normalize();
					if (!isBoundary(j)) 
						_pos[j] += D / 2;
					dx -= D / 2;
				}
			}

			if (!isBoundary(i)) 
				_pos[i] += dx;
		}
	}

	/*
	 *Cette méthode est relativement simple à paralléliser car chaque particule est traitée indépendamment.
	 *Cependant, il faut faire attention à l'ajout dans _pidxInGrid car plusieurs threads pourraient essayer de modifier la même cellule en même temps.
	 */
	void buildCellsNeighborhoud_parallel() {
		_pidxInGrid.clear();
		_pidxInGrid.resize(resX() * resY());
		#pragma omp parallel for
		for (int i = 0; i < particleCount(); ++i) {
			int cellX = static_cast<int>(_pos[i].x);
			int cellY = static_cast<int>(_pos[i].y);

			if (cellX >= 0 && cellX < _resX && cellY >= 0 && cellY < _resY) {
				#pragma omp critical
				_pidxInGrid[idx1d(cellX, cellY)].push_back(i);
			}
		}
	}
	std::vector<tIndex> getNeighbors_parallel(tIndex particleIndex) {
		std::vector<tIndex> neighbors;
		const Vec2f& pos = _pos[particleIndex];
		const Real supportRadiusSquared = _kernel.supportRadius() * _kernel.supportRadius();
		int MAX_NEIGHBORS = 100; //1 cells can approximately contain 5 by 5 particle. Multiple by 9 for a block of neighbour
		neighbors.reserve(MAX_NEIGHBORS);  // Estimation de la taille
		int cellX = static_cast<int>(pos.x);
		int cellY = static_cast<int>(pos.y);
		#pragma omp parallel for collapse(2) // Fusionner les deux boucles for imbriquées
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				int neighborCellX = cellX + i;
				int neighborCellY = cellY + j;

				if (neighborCellX >= 0 && neighborCellX < _resX && neighborCellY >= 0 && neighborCellY < _resY) {
					const std::vector<tIndex>& cell = _pidxInGrid[idx1d(neighborCellX, neighborCellY)];
					#pragma omp parallel for
					for (int k = 0; k < cell.size(); ++k) {
						tIndex neighborIndex = cell[k];
						Vec2f diff = pos - _pos[neighborIndex];
						if (neighborIndex != particleIndex && diff.lengthSquare() < supportRadiusSquared) {
							neighbors.push_back(neighborIndex);
						}
					}
					
				}
			}
		}
		return neighbors;
	}
	void computeAllNeighbors_parallel() {
		_neighborsOf.clear();
		_neighborsOf.resize(particleCount() );

		#pragma omp parallel for
		for (tIndex i = 0; i < particleCount(); ++i) {
			std::vector<tIndex> neighbors = getNeighbors_parallel(i); // Calcul local des voisins
			#pragma omp critical
			_neighborsOf[i] = neighbors; // Mise à jour sécurisée de la structure partagée
			//_neighborsOf.push_back(neighbors);
		}
	}

	void  computeAllNeighborsSimple_viscoelastic() {
		_neighborsOf.clear();
		_neighborsOf.resize(particleCount());

		for (tIndex i = 0; i < particleCount(); ++i) {
			std::vector<tIndex> neighbors;
			for (tIndex j = 0; j < particleCount(); j++)
			{
				Vec2f r_ij = _pos[i] - _pos[j];
				if (std::fabsf(r_ij.length()) < _hVisco )
				{
					neighbors.push_back(j);
				}
			}
			_neighborsOf[i] = neighbors;
		}
	}


	/**
	 *Cette méthode peut également être parallélisée. Chaque particule calcule sa propre densité et pression, donc il n'y a pas de conflit entre les threads.
	 */
	void computePressureDensity_parallel() {

		#pragma omp parallel for
		for (tIndex i = 0; i < particleCount(); ++i) {
			Vec2f r_ij = _pos[i] - _pos[i];  // Auto-influence
			Real influence = _kernel.w(r_ij);
			Real density = _m0 * influence;
			//for (const tIndex& j : getNeighbors_parallel(i)) {
			for (const tIndex& j : _neighborsOf[i]) {
					
				r_ij = _pos[i] - _pos[j];
				influence = _kernel.w(r_ij);
				density += _m0 * influence;
			}

			_d[i] = density;
			_p[i] = std::max(_k * (pow((density / _d0), _gamma) - 1.0f), 0.f);
		}
	}

	void parallelApplyForcesAndComputePosition()
	{
#ifndef PARTICLES_AS_BOUNDARIES
		std::vector<tIndex> need_res; // for collision
#endif
		//std::vector<std::vector<tIndex>> local_leakedParticles(omp_get_max_threads());
		std::vector<std::vector<tIndex>> local_leakedParticles(particleCount());
		#pragma omp parallel
		{
			//int thread_num = omp_get_thread_num();
			//std::vector<tIndex>& my_leakedParticles = local_leakedParticles[thread_num];
			Vec2f accel, fpressure, fvisco;

			#pragma omp for nowait
			for (int i = 0; i < particleCount(); i++) {

				accel = Vec2f(0, 0);
				fvisco = Vec2f(0, 0);
				fpressure = Vec2f(0, 0);
				Vec2f r_ij = _pos[i] - _pos[i];  // Auto-influence
				Vec2f u_ij = _vel[i] - _vel[i]; // Auto-influence
				Vec2f gradW = _kernel.grad_w(r_ij);
				/*fpressure = gradW *  2 * ((_p[i] / (_d[i] * _d[i])));
				Real denom = r_ij.dotProduct(r_ij) + (0.01 * _h * _h);
				fvisco = ((_m0 / _d[i])) * u_ij * (r_ij.dotProduct(gradW) / denom);*/
				//density et pressure
				Real influence = _kernel.w(r_ij);
				Real density = _m0 * influence;
				for (const tIndex& j : _neighborsOf[i]) {

					r_ij = _pos[i] - _pos[j];
					influence = _kernel.w(r_ij);
					density += _m0 * influence;
				}

				_d[i] = density;
				_p[i] = std::max(_k * (pow((density / _d0), _gamma) - 1.0f), 0.f);

#ifdef PARTICLES_AS_BOUNDARIES
				if (!isBoundary(i)) {
#endif
					//for (const tIndex& j : getNeighbors_parallel(i)) {
					for (const tIndex& j : _neighborsOf[i]) {
						r_ij = _pos[i] - _pos[j];
						Vec2f u_ij = _vel[i] - _vel[j];
						gradW = _kernel.grad_w(r_ij);
						//pressure
						fpressure += gradW * ((_p[i] / (_d[i] * _d[i])) + (_p[j] / (_d[j] * _d[j])));

						//Viscosity
						// avoid to divide by 0
						Real denom = r_ij.dotProduct(r_ij) + (0.01 * _h * _h);
						if (denom != 0.0f) {
							fvisco += ((_m0 / _d[j])) * u_ij * (r_ij.dotProduct(gradW) / denom);
						}
					}
#ifdef PARTICLES_AS_BOUNDARIES
				}
#endif


#ifdef PARTICLES_AS_BOUNDARIES
				if (!isBoundary(i)) {// update position if not a boundary
					accel += _g - fpressure + (2.0 * _nu * fvisco);
					//update velocity

//#pragma omp critical
					{
						_vel[i] += _dt * accel;
						_pos[i] += _dt * _vel[i];
					}



#ifdef ADAPTATIVE_TIME
					//update max velocity (for _dt adapatation )
					if (_vel[i].length() > _maxVel)
						_maxVel = _vel[i].length() ;



					if (checkLeak(i) ) {

						bool isAlreadyLeaked = false;
						for (tIndex leak : leakedParticles)
						{
							if (leak == i) {
								isAlreadyLeaked = true;

							}
						}
						if (!isAlreadyLeaked) {
							leakedParticles.push_back(i);
							std::cout << "A leak happened - Number of lost particle  : " << getLeakNumber() << std::endl;

						}

					}
#endif
				}

				
				/*if (checkLeak(i) && !isBoundary(i)) {
					bool isAlreadyLeaked = false;
					for (tIndex leak : my_leakedParticles)
					{
						if (leak == i) 
							isAlreadyLeaked = true;
					}

					if (!isAlreadyLeaked) {
						my_leakedParticles.push_back(i);
						std::cout << "A leak happened - Number of lost particle  : " << getLeakNumber() << std::endl;

					}
				}


				for (const auto& thread_leaks : local_leakedParticles) {
					for (tIndex leak : thread_leaks) {
						leakedParticles.push_back(leak);
					}
				}*/
#endif

#ifndef PARTICLES_AS_BOUNDARIES
				_pos[i] += _dt * _vel[i];
				//collision gesture
				if (_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t)
					need_res.push_back(i);
				for (
					std::vector<tIndex>::const_iterator it = need_res.begin();
					it < need_res.end();
					++it) {
					const Vec2f p0 = _pos[*it];
					_pos[*it].x = clamp(_pos[*it].x, _l, _r);
					_pos[*it].y = clamp(_pos[*it].y, _b, _t);
					_vel[*it] = (_pos[*it] - p0) / _dt;
				}
#endif


				//update colors
				_col[i * 4 + 0] = 0.6;
				_col[i * 4 + 1] = 0.6;
				_col[i * 4 + 2] = _d[i] / _d0;



				//update Velocity lines
				if (gShowVel) {
					_vln[i * 4 + 0] = _pos[i].x;
					_vln[i * 4 + 1] = _pos[i].y;
					_vln[i * 4 + 2] = _pos[i].x + _vel[i].x;
					_vln[i * 4 + 3] = _pos[i].y + _vel[i].y;
				}
				
			}
			
		}
	
	}


	void buildCellsNeighborhoud() {
		// Initialisation : nettoyer les anciennes données
		_pidxInGrid.clear();
		_pidxInGrid.resize(resX() * resY());

		// assign cell to each particle
		for (int i = 0; i < particleCount(); ++i) {
			int cellX = static_cast<int>(_pos[i].x);
			int cellY = static_cast<int>(_pos[i].y);

			if (cellX >= 0 && cellX < (_resX) && cellY >= 0 && cellY < (_resY)) {
				_pidxInGrid[idx1d(cellX, cellY)].push_back(i);

			}
		}
	}

	void computeAllNeighbors() {
		_neighborsOf.clear();
		_neighborsOf.resize(particleCount());

		for (tIndex i = 0; i < particleCount(); ++i) {
			std::vector<tIndex> neighbors = getNeighbors(i); // Calcul local des voisins
			_neighborsOf[i] = neighbors; // Mise à jour sécurisée de la structure partagée
			//_neighborsOf.push_back(neighbors);
		}
	}
	void computePressureDensity()
	{
		for (tIndex i = 0; i < particleCount(); ++i) {
			//density compute
			Vec2f r_ij = _pos[i] - _pos[i];  // Distance entre la particule i et j
			Real influence = _kernel.w(r_ij);
			Real density = _m0 * influence;
			std::vector<tIndex> neigh = getNeighbors(i);
			for (const tIndex& j : neigh) {
				Vec2f r_ij = _pos[i] - _pos[j];  // Distance entre la particule i et j
				Real influence = _kernel.w(r_ij);
				density += _m0 * influence;
			}
			//density compute
			_d[i] = density;

			//pressure compute
			_p[i] = std::max(_k * ((float)pow((density / _d0), _gamma) - 1.0f), 0.0f);
		}



	}
	void applyForcesAndComputePosition()
	{
#ifndef PARTICLES_AS_BOUNDARIES
		std::vector<tIndex> need_res; // for collision
#endif

		int thread_num = omp_get_thread_num();
		Vec2f accel, fpressure, fvisco;
		for (int i = 0; i < particleCount(); i++) {
			accel = Vec2f(0, 0);
			fpressure = Vec2f(0, 0);
			fvisco = Vec2f(0, 0);

#ifdef PARTICLES_AS_BOUNDARIES
			if (!isBoundary(i)) {
#endif
				std::vector<tIndex> neigh = getNeighbors(i);
				for (const tIndex& j : neigh) {
					Vec2f r_ij = _pos[i] - _pos[j];
					Vec2f u_ij = _vel[i] - _vel[j];
					Vec2f gradW = _kernel.grad_w(r_ij);
					//pressure
					fpressure += gradW * ((_p[i] / (_d[i] * _d[i])) + (_p[j] / (_d[j] * _d[j])));

					//Viscosity
					// avoid to divide by 0
					Real denom = r_ij.dotProduct(r_ij) + (0.01 * _h * _h);
					if (denom != 0.0f) {
						fvisco += ((_m0 / _d[j])) * u_ij * (r_ij.dotProduct(gradW) / denom);
					}
				}
#ifdef PARTICLES_AS_BOUNDARIES
			}
#endif
#ifndef PARTICLES_AS_BOUNDARIES
		accel += _g - fpressure + (2.0 * _nu * fvisco);
		//update velocity

		_vel[i] += _dt * accel;
		//update position 
		_pos[i] += _dt * _vel[i];
#endif

#ifdef PARTICLES_AS_BOUNDARIES
			if (!isBoundary(i)) {// update position if not a boundary
				accel += _g - fpressure + (2.0 * _nu * fvisco);
				//update velocity
				
				_vel[i] += _dt * accel;
				//update position 
				_pos[i] += _dt * _vel[i];
				
#ifdef ADAPTATIVE_TIME
				//update max velocity (for _dt adapatation )
				if (_vel[i].length() > _maxVel)
					_maxVel = _vel[i].length();
#endif
			}

			if (checkLeak(i) && !isBoundary(i)) {

				bool isAlreadyLeaked = false;
				for (tIndex leak : leakedParticles)
				{
					if (leak == i){
						isAlreadyLeaked = true;

					}
				}
				if (!isAlreadyLeaked) {
					leakedParticles.push_back(i);
					std::cout << "A leak happened - Number of lost particle  : " << getLeakNumber() << std::endl;

				}

			}


#endif




#ifndef PARTICLES_AS_BOUNDARIES
			_pos[i] += _dt * _vel[i];
			//collision gesture
			if (_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t)
				need_res.push_back(i);
			for (
				std::vector<tIndex>::const_iterator it = need_res.begin();
				it < need_res.end();
				++it) {
				const Vec2f p0 = _pos[*it];
				_pos[*it].x = clamp(_pos[*it].x, _l, _r);
				_pos[*it].y = clamp(_pos[*it].y, _b, _t);
				_vel[*it] = (_pos[*it] - p0) / _dt;
			}
#endif

#ifndef VISCOELASTIC

			//update colors
			_col[i * 4 + 0] = 0.6;
			_col[i * 4 + 1] = 0.6;
			_col[i * 4 + 2] = _d[i] / _d0;



			//update Velocity lines
			if (gShowVel) {
				_vln[i * 4 + 0] = _pos[i].x;
				_vln[i * 4 + 1] = _pos[i].y;
				_vln[i * 4 + 2] = _pos[i].x + _vel[i].x;
				_vln[i * 4 + 3] = _pos[i].y + _vel[i].y;
			}
#endif

		}

	}
	
	

	std::vector<tIndex> getNeighbors(tIndex particleIndex) {
		std::vector<tIndex> neighbors;
		const Vec2f& pos = _pos[particleIndex];

		int cellX = static_cast<int>(pos.x);
		int cellY = static_cast<int>(pos.y);
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				int neighborCellX = cellX + i;
				int neighborCellY = cellY + j;

				if (neighborCellX >= 0 && neighborCellX < _resX && neighborCellY >= 0 && neighborCellY < _resY) {
					const std::vector<tIndex>& cell = _pidxInGrid[idx1d(neighborCellX, neighborCellY)];
					for (tIndex neighborIndex : cell) {
						if (neighborIndex != particleIndex && (pos - _pos[neighborIndex]).length() < _kernel.supportRadius()) {
							neighbors.push_back(neighborIndex);
							//_neighborsOf[idx1dnei(cellX, cellY)].push_back(neighborIndex);
						}
					}
				}
			}
		}
		return neighbors;
	}

	/*void computeAllNeighbors() {
		_neighborsOf.clear();
		_neighborsOf.resize(particleCount()*MAX_NEIGHBORS);

		for (tIndex i = 0; i < particleCount(); ++i) {
			auto neighbors = getNeighbors(i);
			//_neighborsOf.push_back(neighbors);
			//_neighborsOf[i] = neighbors; 
		}
	}*/


	bool isBoundary(const tIndex& p)
	{
#ifdef PARTICLES_AS_BOUNDARIES
		return (p >=( particleCount() - _particleBoundariesNumber));
#endif

#ifndef PARTICLES_AS_BOUNDARIES
	return false;
#endif


	}

	bool checkLeak(const tIndex& i)
	{
		if(_pos[i].x<(_l * 2.0f) || _pos[i].y<(_b * 2.0f) || _pos[i].x>(_r * 2.0f) || _pos[i].y>(_t * 2.0f))
			return true;
		return false;
		
	}
	int getLeakNumber() { return leakedParticles.size(); }

	void resolveCollisionBoundary(const tIndex& boundary, const tIndex& j) {

		Vec2f r_ij = _pos[boundary] - _pos[j];  // Distance entre la particule boundary et j. r_ij.x = dx, r_ij =dy
		Real distance = r_ij.length();

		if (!checkCollision(distance)) return; //if no collision return

		Real radius = _h/2 ;
		// collision direction
		Real nx = r_ij.x / distance;
		Real ny = r_ij.y / distance;

		Real relativeVelocityX = _vel[boundary].x - _vel[j].x;
		Real relativeVelocityY = _vel[boundary].y - _vel[j].y;

		// relative velocity in the direction of collision
		Real relativeSpeed = relativeVelocityX * nx + relativeVelocityY * ny;

		if (relativeSpeed > 0) return;// if particle go away from each other there is no collision so return


		//   restitution factor ( elasticity )
		Real restitution = 0.15; // factor to adjust


		// impulse of collision
		Real impulsion = -(1.0 + restitution) * relativeSpeed;
		impulsion /= ((1.0 /radius) + (1.0 / radius));

		// apply impulsion

		_vel[j].x -= impulsion * nx / radius;
		_vel[j].y -= impulsion * ny / radius;


		// Update position to avoid overlap
		Real overlap = 0.5* (distance - _h); // overlap to apply to avoid inter particle penetration
		
		_pos[j].x += overlap * nx;
		_pos[j].y += overlap * ny;
		
		
	}
	bool checkCollision(Real distance)
	{
		return (distance < (_h));
	}


	
	

	void computeDensity()
	{
	for(tIndex i = 0; i < particleCount(); ++i) {
	  Real density = 0.0;
	  for (const tIndex& j : getNeighbors(i)) {
	    Vec2f r_ij = _pos[i] - _pos[j];  // Distance entre la particule i et j
	    Real distance = r_ij.length();
        Real influence = _kernel.w(r_ij);
        density += _m0 * influence;
	    
	  }
	  _d[i] = density;
	}
	}

	void computePressure()
	{
	for(int i=0; i< particleCount(); i++){
	  _p[i] = std::max(_k*((float)pow((_d[i]/_d0),_gamma)- 1.0f), 0.0f);
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
	  Vec2f f(0.f,0.f);
	  for (const tIndex& j : getNeighbors(i)) {
	    Vec2f r_ij = _pos[i] - _pos[j];  // Distance entre la particule i et j
	    Real distance = r_ij.length();
	    if (distance < _kernel.supportRadius()) {
	        f+=_kernel.grad_w(r_ij)*_m0*((_p[i]/(_d[i]*_d[i]))+(_p[j]/(_d[j]*_d[j])));
	    }
	  }
	  _acc[i] += -f/_m0;
	}
	}

	void applyViscousForce()
	{

	for (int i = 0; i < particleCount(); ++i) {
	  Vec2f f(0.0, 0.0);

	  for (const tIndex& j : getNeighbors(i)) {
	    Vec2f x_ij = _pos[i] - _pos[j];
	    Vec2f u_ij = _vel[i] - _vel[j];
	    Vec2f gradW = _kernel.grad_w(x_ij);

	    // Éviter la division par zéro et assurer la stabilité numérique
	    Real denom = x_ij.dotProduct(x_ij) +  (0.01 * _h * _h);
	    if (denom > 0.0f) {
	        f += ( (_m0 / _d[j]))*u_ij * (x_ij.dotProduct(gradW) / denom);
	    }
	  }

	  _acc[i] += 2.0 * _nu  *f;
	}
	}

	void updateVelocity()
	{
	for (int i = 0; i < particleCount(); ++i) {
	  _vel[i] += _dt * _acc[i];
	}
	}

	void updatePosition()
	{
	for (int i = 0; i < particleCount(); ++i) {
	  _pos[i] += _dt * _vel[i];
	}
	}

	// simple collision detection/resolution for each particle
	void resolveCollisionBoundary()
	{
	std::vector<tIndex> need_res;
	for(tIndex i=0; i<particleCount(); ++i) {
	  if(_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t)
	    need_res.push_back(i);
	}

	for(
	  std::vector<tIndex>::const_iterator it=need_res.begin();
	  it<need_res.end();
	  ++it) {
	  const Vec2f p0 = _pos[*it];
	  _pos[*it].x = clamp(_pos[*it].x, _l, _r);
	  _pos[*it].y = clamp(_pos[*it].y, _b, _t);
	  _vel[*it] = (_pos[*it] - p0)/_dt;
	  }
	}

	void updateColor()
	{
		for (tIndex i = 0; i < particleCount(); ++i) {

			_col[i * 4 + 0] = 0.6;
			_col[i * 4 + 1] = 0.6;
			_col[i * 4 + 2] = _d[i] / _d0;

		}
	}

	void updateVelLine()
	{
	for(tIndex i=0; i<particleCount(); ++i) {
	  _vln[i*4+0] = _pos[i].x;
	  _vln[i*4+1] = _pos[i].y;
	  _vln[i*4+2] = _pos[i].x + _vel[i].x;
	  _vln[i*4+3] = _pos[i].y + _vel[i].y;
		}
	}


	inline tIndex idx1d(const int& i, const int& j) { return i + j*resX(); }
	inline tIndex idx1dnei(const int& i, const int& j) { return i + j * MAX_NEIGHBORS; }

  const CubicSpline _kernel;

  // particle data
  std::vector<Vec2f> _pos;      // position
  std::vector<Vec2f> _vel;      // velocity
  std::vector<Vec2f> _acc;      // acceleration
  std::vector<Real>  _p;        // pressure
  std::vector<Real>  _d;        // density
  std::vector<tIndex> leakedParticles; //lost particles

  std::vector< std::vector<tIndex> > _pidxInGrid; // will help you find neighbor particles
  std::vector<std::vector<tIndex>> _neighborsOf;

  std::vector<float> _col;    // particle color; just for visualization
  std::vector<float> _vln;    // particle velocity lines; just for visualization

  // simulation
  Real _dt;                     // time step

  int _resX, _resY;             // background grid resolution
  int MAX_NEIGHBORS = 4*4*9; //1 cells can approximately contain 4 by 4 particle. Multiple by 9 for a block of neighbour

  tIndex _particleBoundariesNumber;
  // wall
  Real _l, _r, _b, _t;          // wall (boundary)
  Real boundaryOffset;
 



  // SPH coefficients
  Real _nu;                     // viscosity coefficient
  Real _d0;                     // rest density
  Real _h;                      // particle spacing (i.e., diameter)
  Vec2f _g;                     // gravity

  Real _m0;                     // rest mass
  Real _k;                      // EOS coefficient

  Real _eta;
  Real _c;                      // speed of sound
  Real _gamma;                  // EOS power factor

	//For _dt integration
  Real _maxVel;
  Real isFirstStep = true;


	//viscoelastic
  Real _kViscoElas;					//same function as _k
	Real _kViscoElasNear;					//EOS near
	Real _d0ViscoELas;				//_d0 for visco elastic sim
  std::vector<Real>  _dNear;	//density near
  std::vector<Real>  _pNear;	//pressure near
  Real _hVisco;					// h for viscofluid

  Real _sigma ;					// viscosity factor ( the high it is the more viscous the fluid would be)
  Real _beta;					// quadratic dependance compared with vvelocity. Usefull to avoid particle interpenetration by eliminating high intern speed. SHpuld be non nul
  std::vector<Vec2f> _posPrevious;      // position
  //Real _L[1000][1000];			//spring length value between two fluid particles
  std::vector< std::vector<Real> > _L;			//spring length value between two fluid particles
  Real _L0;						// spring rest length
  Real _k_spring;				//spring constant
  Real _alpha;					//plasticity constant
  Real _gammaSpring;
  bool _applyGravity;
  bool _applyViscosity;
  bool _applySprings;
  bool _SPHfluid;

};

SphSolver gSolver(0.08,0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);



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
void windowSizeCallback(GLFWwindow *window, int width, int height)
{
  gWindowWidth = width;
  gWindowHeight = height;
  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void initImGUI()
{
	dtGUI = gSolver.getDt();
	d0GUI = gSolver.getD0ViscoElas();
	_kGUI = gSolver.getKViscoElas();
	_kNearGUI = gSolver.getKNearViscoElas();
	_k_springGUI = gSolver.getKSpring();

	_hViscoGUI = gSolver.getHViscoElas();
	_L0GUI = gSolver.getL0();
	_alphaGUI = gSolver.getAlpha();
	_betaGUI = gSolver.getBeta();

	_sigmaGUI = gSolver.getSigma();

	_gammaSpringGUI = gSolver.getGammaSpring();
	gApplyGravity = gSolver.isGravityApplied();
	gApplyVisco = gSolver.isViscosityApplied();
	gApplySprings = gSolver.isSpringsApplied();
	gSPHfluid = gSolver.isSPHApplied();


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
	gSolver.initScene(48, 32, 32, 16);

}
// Executed each time a key is entered.
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
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
}
void initGLFW()
{
  // Initialize GLFW, the library responsible for window management
  if(!glfwInit()) {
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
  gWindowWidth = gSolver.resX()*kViewScale;
  gWindowHeight = gSolver.resY()*kViewScale;
  gWindow = glfwCreateWindow(
    gSolver.resX()*kViewScale, gSolver.resY()*kViewScale,
    "Basic SPH Simulator", nullptr, nullptr);
  if(!gWindow) {
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

  std::cout << "Window created: " <<
    gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void exitOnCriticalError(const std::string &message)
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
  if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    exitOnCriticalError("[Failed to initialize OpenGL context]");

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init()
{
  gSolver.initScene(48, 32, 32, 16);
  initGLFW();                   // Windowing system
  initOpenGL();
#ifdef IMGUI
  initImGUI();
#endif
}

void clear()
{
  glfwDestroyWindow(gWindow);
  // Nettoyage
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwTerminate();
}

// The main rendering call
void render()
{
  glClearColor(.4f, .4f, .4f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#ifdef IMGUI

  // Commencer la nouvelle frame ImGui
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
#endif

  // grid guides
  if(gShowGrid) {
    glBegin(GL_LINES);
    for(int i=1; i<gSolver.resX(); ++i) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), 0.0);
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
    }
    for(int j=1; j<gSolver.resY(); ++j) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(0.0, static_cast<Real>(j));
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
    }
    glEnd();
  }

  // render particles
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glPointSize(0.25f*kViewScale);

  glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
  glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
  glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // velocity
  if(gShowVel) {
    glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
    glDrawArrays(GL_LINES, 0, gSolver.particleCount()*2);

    glDisableClientState(GL_VERTEX_ARRAY);

  }

#ifdef IMGUI
  // Ici, construisez votre interface ImGui
  ImGui::Begin("Debug ImGui de Adama");
  // Checkbox that appears in the window
  ImGui::Checkbox("saveFile", &gSaveFile);
  ImGui::Checkbox("show grid", &gShowGrid);
  ImGui::Checkbox("show Velocities", &gShowVel);
  ImGui::Checkbox("Apply gravity", &gApplyGravity);
  ImGui::Checkbox("Apply Viscosity", &gApplyVisco);
  ImGui::Checkbox("Apply Springs", &gApplySprings);
  ImGui::Checkbox("SPH fluid", &gSPHfluid);

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

  ImGui::SliderFloat("d0", &d0GUI, 0.f, 2000.f,"Valeur: % .3f", 1);
  //ImGui::InputFloat(" ", &d0GUI, 0.1f, 1.0f, "%.3f");

  ImGui::SliderFloat("dt", &dtGUI, 0.00004f, 1.0f, "Valeur: %.3f", 0.00001f);
  //ImGui::InputFloat(" ", &dtGUI, 0.1f, 1.0f, "%.3f");

  ImGui::SliderFloat("alpha", &_alphaGUI, 0.0001f, 10.0f, "Valeur: %.3f", 0.001f);
  //ImGui::InputFloat(" ", &_alphaGUI, 0.1f, 1.0f, "%.3f");

  ImGui::SliderFloat("beta", &_betaGUI, 0.0005f, 10.0f, "Valeur: %.3f", 0.001f);
  //ImGui::InputFloat(" ", &_betaGUI, 0.1f, 1.0f, "%.3f");


  // Fancy color editor that appears in the window
  //ImGui::ColorEdit4("Color", color);
  ImGui::End();

  // Rendu d'ImGui
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif

  if(gSaveFile) {
    std::stringstream fpath;
    fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

    std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
    const short int w = gWindowWidth;
    const short int h = gWindowHeight;
    std::vector<int> buf(w*h*3, 0);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

    FILE *out = fopen(fpath.str().c_str(), "wb");
    short TGAhead[] = {0, 2, 0, 0, 0, 0, w, h, 24};
    fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    fwrite(&(buf[0]), 3*w*h, 1, out);
    fclose(out);
    gSaveFile = false;

    std::cout << "Done" << std::endl;
  }
}

// Update any accessible variable based on the current time
void update(const float currentTime)
{
	if(!gAppTimerStoppedP) {
		// NOTE: When you want to use application's dt ...
		/*
		const float dt = currentTime - gAppTimerLastClockTime;
		gAppTimerLastClockTime = currentTime;
		gAppTimer += dt;
		*/

		//save a pic after n step

		
#ifdef SAVEIMAGES
		int n;
		n= gSolver.getLoopNum();// 50;
		// solve 10 steps for better stability ( chaque step est un pas de temps )
		for (int i = 0; i < n; ++i)
#endif
#ifdef IMGUI
			gSolver.updateFactors(dtGUI, d0GUI, _kGUI, _kNearGUI, _k_springGUI,
				_hViscoGUI, _L0GUI, _alphaGUI, _betaGUI, _sigmaGUI, _gammaSpringGUI);
		gSolver.applyGravity(gApplyGravity);
		gSolver.applyViscosity(gApplyVisco);
		gSolver.applySprings(gApplySprings);
		gSolver.applySPH(gSPHfluid);
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
}

int main(int argc, char **argv)
{
  init();
  while(!glfwWindowShouldClose(gWindow)) {
    update(static_cast<float>(glfwGetTime()));
    render();
    glfwSwapBuffers(gWindow);
    glfwPollEvents();
  }
  clear();
  std::cout << " > Quit" << std::endl;
  return EXIT_SUCCESS;
}

