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

#define  FASTVERSION //quicker version by merging some function
#define  PARTICLES_AS_BOUNDARIES //work only if fatsversion defined
#define ADAPTATIVE_TIME //work only if fatsversion defined


#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.141592
#endif
#ifndef ADAPTATIVE_TIME
#define NUM_BOUNDARY_LAYER 2
#endif

#ifdef ADAPTATIVE_TIME
#define NUM_BOUNDARY_LAYER 3
#endif

#include "Vector.hpp"
typedef Vec2f vec;
// window parameters
GLFWwindow *gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline {
public:
  explicit CubicSpline(const Real h=1) : _dim(2)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const Real h)
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
  private :
    const bool _boundary;
    Vec2f _position, _velocity, _acc;
    Real _pressure, _density;
  public:
	Particle( Vec2f position, bool boundary):_position(position),_boundary(boundary){}

	Vec2f getPosition()const{return _position;}
	void setPosition(const Vec2f& position){_position = position;}

	const bool isBoundary()const{return  _boundary;}

	void setVelocity(const Vec2f& velocity){_velocity = velocity;}
    Vec2f getVelocity()const { return _velocity; }

	void setAcc(const Vec2f& acc) { _acc = acc; }
    Vec2f getAcc()const { return _acc; }

	//maybe useless
    void setPressure(const Real pressure) { _pressure = pressure; }
    Real getPressure()const { return _pressure; }

    void setDensity(const Real density) { _density = density; }
    Real getDensity() const { return _density; }




};

class SphSolver {
public:
  explicit SphSolver(
    const Real nu=0.01, const Real h=0.5, const Real density=1e3,
    const Vec2f g=Vec2f(0, -9.8), const Real eta=0.01, const Real gamma=7.0) :
    _kernel(h), _nu(nu),_h(h), _d0(density),
    _g(g), _eta(eta), _gamma(gamma)
  {
	  _dt = 0.0005f;

#ifdef ADAPTATIVE_TIME
	  _dt = 0.f;
#endif

    _m0 = _d0*_h*_h;
    _c = std::fabs(_g.y)/_eta;
    _k = _d0*_c*_c/_gamma;
	_maxVel = std::fabs(_g.length());;
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

  	// set wall for boundary
    _l = 0.5*_h - boundaryOffset;
    _r = static_cast<Real>(res_x) - 0.5*_h + boundaryOffset;
    _b = 0.5*_h - boundaryOffset;
    _t = static_cast<Real>(res_y) - 0.5*_h + boundaryOffset;


    // sample a fluid mass
    for(int j=0; j<f_height; ++j) {
      for(int i=0; i<f_width; ++i) {
        _pos.push_back(Vec2f(i+0.25f+2*_h, j+0.25f+2*_h));
        _pos.push_back(Vec2f(i+0.75f+2*_h, j+0.25f+2*_h));
        _pos.push_back(Vec2f(i+0.25f+2*_h, j+0.75f+2*_h));
        _pos.push_back(Vec2f(i+0.75f+2*_h, j+0.75f+2*_h));
      }
    }
	Real p = _h/2.f;

#ifdef PARTICLES_AS_BOUNDARIES
	_particleBoundariesNumber = 0;
	//build the bondaries with particles//fill the horizontal boundaries
	for (int i = _l+1; i < res_x; i++) {
		//bottom line
		if (i < _r - 1) {


			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(i + (p), _b + (_h * j)));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(i + (3 * p), _b + (_h * j)));
				_particleBoundariesNumber++;
			}
	}
        
		//top line
        if(i<_r-1){

			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(i + (p), (_t - (_h * j))));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(i + (3 * p), (_t- (_h * j))));
				_particleBoundariesNumber++;
			}

        }
	}
	//build the bondaries with particles//fill the vertical boundaries
    for (int i = _b; i < res_y;i++) {
		//bottom line
        if (i < _t) {

			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(_l + (_h * j), i + (p)));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(_l+ (_h * j), i + (3 * p)));
				_particleBoundariesNumber++;
			}


        }
		//top line
        if (i < _t) {

			for (int j = 0; j < NUM_BOUNDARY_LAYER; j++) {
				_pos.push_back(Vec2f(_r - (_h * j), i + (p)));
				_particleBoundariesNumber++;
				_pos.push_back(Vec2f(_r - (_h * j), i + (3 * p)));
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

    updateColor();
	std::cout<<"Total Number of particle : "<<particleCount() << " | Number of non boundary particles : " << particleCount() - _particleBoundariesNumber << " |  Number of boundary particles : " << _particleBoundariesNumber << std::endl;

  }

  void update()
  {

    std::cout << '.' << std::flush;
	
  	buildCellsNeighborhoud();
#ifndef FASTVERSION
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
#ifdef FASTVERSION
	
  	computeStates();
	applyForcesAndComputePosition();

#ifdef ADAPTATIVE_TIME
	/*if(isFirstStep)
	{
	_maxVel += std::fabs(_g.length());
		isFirstStep = false;
	}
  	else
	{
		//_maxVel += _dt * std::fabs(_g.length());// / (_h * 0.4f);
	}*/
	_dt = (0.09f * _h) / _maxVel; //CFL condition
	//std::cout << "Vmax=" << _maxVel <<" dt="<<_dt<< std::flush;

	_maxVel = 0.f;
#endif


#endif

  }

  tIndex particleCount() const { return _pos.size(); }
  const Vec2f& position(const tIndex i) const { return _pos[i]; }
  const float& color(const tIndex i) const { return _col[i]; }
  const float& vline(const tIndex i) const { return _vln[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }
  Real getLoopNum()
  {
	  if (isFirstStep){
		  isFirstStep = false;
		  return 10;
	  }

	  std::cout << " loop num = " << static_cast<int> (0.01 / _dt) << std::endl;
	  return static_cast<int> (0.1 / _dt);
  }


private:
	void computeStates()
	{
		for (tIndex i = 0; i < particleCount(); ++i) {
			//density compute
			Vec2f r_ij = _pos[i] - _pos[i];  // Distance entre la particule i et j
			Real influence = _kernel.w(r_ij);
			Real density = _m0 * influence;
			for (const tIndex& j : getNeighbors(i)) {
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
		std::vector<tIndex> need_res; // for collision 
		Vec2f accel;
		Vec2f fpressure;
		Vec2f fvisco;
		for (int i = 0; i < particleCount(); i++) {
			accel = Vec2f(0, 0);
			fpressure = Vec2f(0, 0);
			fvisco = Vec2f(0, 0);

#ifdef PARTICLES_AS_BOUNDARIES
			if(!isBoundary(i)){
#endif
				for (const tIndex& j : getNeighbors(i)) {
					Vec2f r_ij = _pos[i] - _pos[j];
					Vec2f u_ij = _vel[i] - _vel[j];
					Vec2f gradW = _kernel.grad_w(r_ij);
					//pressure
					fpressure += gradW  * ((_p[i] / (_d[i] * _d[i])) + (_p[j] / (_d[j] * _d[j])));

					//Viscosity
					// avoid to divide by 0
					Real denom = r_ij.dotProduct(r_ij) + (0.01 * _h * _h);
					if (denom != 0.0f) {
						fvisco += ((_m0 / _d[j])) * u_ij *(r_ij.dotProduct(gradW) / denom);
					}
				}
#ifdef PARTICLES_AS_BOUNDARIES
			}
#endif


#ifdef PARTICLES_AS_BOUNDARIES
			if (!isBoundary(i)) {// update position if not a boundary
				accel += _g - fpressure + (2.0 * _nu * fvisco);
				//update velocity
				_vel[i] += _dt * accel;
				//update position 
				
#ifdef ADAPTATIVE_TIME
				//update max velocity (for _dt adapatation )
				if (_vel[i].length() > _maxVel)
					_maxVel = _vel[i].length();

				
#endif
				
				_pos[i] += _dt * _vel[i];
			}

			if (checkLeak(i) && !isBoundary(i)) {
				bool isAlreadyLeaked = false;
				for (tIndex leak : leakedParticles)
				{
					if (leak == i)
						isAlreadyLeaked = true;
				}
				if (!isAlreadyLeaked){
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
	bool isBoundary(const tIndex& p){return (p >=( particleCount() - _particleBoundariesNumber));}

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


	void buildCellsNeighborhoud(){
	// Initialisation : nettoyer les anciennes données
	_pidxInGrid.clear();
	_pidxInGrid.resize(resX()*resY());

	// assign cell to each particle
	for (size_t i = 0; i < particleCount(); ++i) {
	  int cellX = static_cast<int>(_pos[i].x );
	  int cellY = static_cast<int>(_pos[i].y );

	  if (cellX >= 0 && cellX < (_resX ) && cellY >= 0 && cellY < (_resY)) {
	    _pidxInGrid[idx1d(cellX, cellY)].push_back(i);
		
	  }

	}

	}
	std::vector<tIndex> getNeighbors(tIndex particleIndex){

		std::vector<tIndex> neighbors;
		int cellX = static_cast<int>(_pos[particleIndex].x );
		int cellY = static_cast<int>(_pos[particleIndex].y );

		for (int i = -1; i <= 1; ++i) {
		  for (int j = -1; j <= 1; ++j) {
		    int neighborCellX = cellX + i;
		    int neighborCellY = cellY + j;

		    if (neighborCellX >= 0 && neighborCellX < _resX && neighborCellY >= 0 && neighborCellY < _resY) {
		        const std::vector<tIndex>& cell = _pidxInGrid[idx1d(neighborCellX, neighborCellY)];
		        for (tIndex neighborIndex : cell) {
		            if (neighborIndex != particleIndex && (_pos[particleIndex] - _pos[neighborIndex]).length() < _kernel.supportRadius()) {
		                neighbors.push_back(neighborIndex);
		            }
		        }
		    }
		  }
		}
		return neighbors;
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

  const CubicSpline _kernel;

  // particle data
  std::vector<Vec2f> _pos;      // position
  std::vector<Vec2f> _vel;      // velocity
  std::vector<Vec2f> _acc;      // acceleration
  std::vector<Real>  _p;        // pressure
  std::vector<Real>  _d;        // density
  std::vector<tIndex> leakedParticles; //lost particles

  std::vector< std::vector<tIndex> > _pidxInGrid; // will help you find neighbor particles

  std::vector<float> _col;    // particle color; just for visualization
  std::vector<float> _vln;    // particle velocity lines; just for visualization

  // simulation
  Real _dt;                     // time step

  int _resX, _resY;             // background grid resolution

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
};

SphSolver gSolver(0.08,0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);

void printHelp()
{
  std::cout <<
    "> Help:" << std::endl <<
    "    Keyboard commands:" << std::endl <<
    "    * H: print this help" << std::endl <<
    "    * P: toggle simulation" << std::endl <<
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

// Executed each time a key is entered.
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if(action == GLFW_PRESS && key == GLFW_KEY_H) {
    printHelp();
  } else if(action == GLFW_PRESS && key == GLFW_KEY_S) {
    gSaveFile = !gSaveFile;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_G) {
    gShowGrid = !gShowGrid;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_V) {
    gShowVel = !gShowVel;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_P) {
    gAppTimerStoppedP = !gAppTimerStoppedP;
    if(!gAppTimerStoppedP)
      gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
  } else if(action == GLFW_PRESS && key == GLFW_KEY_Q) {
    glfwSetWindowShouldClose(window, true);
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
  gSolver.initScene(48, 32, 16, 16);

  initGLFW();                   // Windowing system
  initOpenGL();
}

void clear()
{
  glfwDestroyWindow(gWindow);
  glfwTerminate();
}

// The main rendering call
void render()
{
  glClearColor(.4f, .4f, .4f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

		  int n = gSolver.getLoopNum();// 50; 
		// solve 10 steps for better stability ( chaque step est un pas de temps )
		for(int i=0; i<n; ++i)
		   gSolver.update();
	

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
