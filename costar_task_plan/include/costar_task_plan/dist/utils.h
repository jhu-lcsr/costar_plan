// This file is part of libgcop, a library for Geometric Control, Optimization, and Planning (GCOP)
//
// Copyright (C) 2004-2014 Marin Kobilarov <marin(at)jhu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef GCOP_UTILS_H
#define GCOP_UTILS_H

#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <iostream>
#include <stdio.h>

namespace gcop {

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
#include <windows.h>
#endif

#ifdef OS_WIN  

#ifndef usleep
#define usleep(x) Sleep(x/1000)
#endif

#endif

#define EPS (1e-10)
#define ISZERO(x) (fabs(x)<EPS)

#ifndef MAX
#define MAX(a,b)((a)>(b)?(a):(b))
#endif
#ifndef MIN
#define MIN(a,b)((a)<(b)?(a):(b))
#endif

#ifndef RND
#define RND (std::rand()/(double)RAND_MAX)
#endif

#define TIME_CMP(a,b) (a.tv_sec != b.tv_sec || a.tv_usec != b.tv_usec)

#ifndef RAD2DEG
#define RAD2DEG(x) (180.0*(x)/M_PI)
#endif
#ifndef DEG2RAD
#define DEG2RAD(x) (M_PI*(x)/180)
#endif

#define SQR(a) ((a)*(a))

#define SIGN(a) ((a) < 0 ? -1 : 1)

//#define DIST(ya,ya,xb,yb) (sqrt(((xa)-(xb))*((xa)-(xb))+((ya)-(yb))*((ya)-(yb))))
//#define NORM(x,y) (sqrt((x)*(x)+(y)*(y)))

#define DIST(a,b) (sqrt(((a)[0]-(b)[0])*((a)[0]-(b)[0])+((a)[1]-(b)[1])*((a)[1]-(b)[1])))
#define NORM(a) (sqrt((a)[0]*(a)[0]+(a)[1]*(a)[1]))
#define MINUS(c,a,b) (c)[0]=(a)[0]-(b)[0];(c)[1]=(a)[1]-(b)[1];
#define PLUS(c,a,b) (c)[0]=(a)[0]+(b)[0];(c)[1]=(a)[1]+(b)[1];
#define DOT(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1])
#define MULT(a,b,n) (a)[0]=(b)[0]*(n);(a)[1]=(b)[1]*(n);
#define DIV(a,b,n) (a)[0]=(b)[0]/(n);(a)[1]=(b)[1]/(n);
#define DET(a,b) ((a)[0]*(b)[1]-(a)[1]*(b)[0])
#define NEG(a,b) (a)[0]=-(b)[0];(a)[1]=-(b)[1];

#define SET(a,b) (a)[0]=(b)[0];(a)[1]=(b)[1];
#define CLEAR(a) (a)[0]=0;(a)[1]=0;

#define DIST3(a,b) (sqrt(SQR((a)[0]-(b)[0])+SQR((a)[1]-(b)[1])+SQR((a)[2]-(b)[2])))
#define NORM3(a) (sqrt((a)[0]*(a)[0]+(a)[1]*(a)[1]+(a)[2]*(a)[2]))
#define MINUS3(c,a,b) (c)[0]=(a)[0]-(b)[0];(c)[1]=(a)[1]-(b)[1];(c)[2]=(a)[2]-(b)[2];
#define PLUS3(c,a,b) (c)[0]=(a)[0]+(b)[0];(c)[1]=(a)[1]+(b)[1];(c)[2]=(a)[2]+(b)[2];
#define AVE3(c,a,b) (c)[0]=((a)[0]+(b)[0])/2;(c)[1]=((a)[1]+(b)[1])/2;(c)[2]=((a)[2]+(b)[2])/2;
#define DOT3(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define MULT3(a,b,n) (a)[0]=(b)[0]*(n);(a)[1]=(b)[1]*(n);(a)[2]=(b)[2]*(n);
#define DIV3(a,b,n) (a)[0]=(b)[0]/(n);(a)[1]=(b)[1]/(n);(a)[2]=(b)[2]/(n);
//#define DET3(a,b) ((a)[0]*(b)[1]-(a)[1]*(b)[0])
#define NEG3(a,b) (a)[0]=-(b)[0];(a)[1]=-(b)[1];(a)[2]=-(b)[2];

#define SET3(a,b) (a)[0]=(b)[0];(a)[1]=(b)[1];(a)[2]=(b)[2];
#define CLEAR3(a) (a)[0]=0;(a)[1]=0;(a)[2]=0;

#define CROSS(c,a,b) (c)[0]=(a)[1]*(b)[2]-(a)[2]*(b)[1]; (c)[1]=-(a)[0]*(b)[2]+(a)[2]*(b)[0]; (c)[2]=(a)[0]*(b)[1]-(a)[1]*(b)[0]; 

#define NORM4(a) (sqrt((a)[0]*(a)[0]+(a)[1]*(a)[1]+(a)[2]*(a)[2]+(a)[3]*(a)[3]))
#define MINUS4(c,a,b) (c)[0]=(a)[0]-(b)[0];(c)[1]=(a)[1]-(b)[1];(c)[2]=(a)[2]-(b)[2];(c)[3]=(a)[3]-(b)[3];
#define PLUS4(c,a,b) (c)[0]=(a)[0]+(b)[0];(c)[1]=(a)[1]+(b)[1];(c)[2]=(a)[2]+(b)[2];(c)[3]=(a)[3]+(b)[3];
#define MULT4(a,b,n) (a)[0]=(b)[0]*(n);(a)[1]=(b)[1]*(n);(a)[2]=(b)[2]*(n);(a)[3]=(b)[3]*(n);
#define DIV4(a,b,n) (a)[0]=(b)[0]/(n);(a)[1]=(b)[1]/(n);(a)[2]=(b)[2]/(n);(a)[3]=(b)[3]/(n);
#define SET4(a,b) (a)[0]=(b)[0];(a)[1]=(b)[1];(a)[2]=(b)[2];(a)[3]=(b)[3];
#define CLEAR4(a) (a)[0]=0;(a)[1]=0;(a)[2]=0;(a)[3]=0;
#define DIST4(a,b) (sqrt(SQR((a)[0]-(b)[0])+SQR((a)[1]-(b)[1])+SQR((a)[2]-(b)[2])+SQR((a)[3]-(b)[3])))

/**
 * Start a timer
 * @param timer timer
 */
inline void timer_start(struct timeval &timer)
{
  gettimeofday(&timer, 0);
}

/**
 * Get elapsed time in microseconds
 * Timer should be started with timer_start(timer)
 * @param timer timer
 * @return elapsed time
 */
inline long timer_us(struct timeval &timer)
{
  struct timeval now;
  gettimeofday(&now, 0);
  return (now.tv_sec - timer.tv_sec)*1000000 + now.tv_usec - timer.tv_usec;
}

/**
 * Normal distribution likelihood
 * @param mu mean
 * @param sigma variance
 * @param x sample
 * @return likelihood
 */
inline double normal_dist(double mu, double sigma, double x) 
{
  return exp(-(x - mu)*(x - mu)/(2*sigma*sigma))/(sigma*sqrt(2*M_PI));
}

/**
 * 0-mean unit-stdev normal distribution sample
 * @return sample
 */
inline double random_normal()
{
  static const double r_max=RAND_MAX;
  double U1,U2,V1,V2;
  double S = 2;
  while(S>=1) {
    U1 = std::rand()/r_max;
    U2 = std::rand()/r_max;
    V1 = 2*U1-1;
    V2 = 2*U2-1;
    S = V1*V1+V2*V2;
  }
  return V1*sqrt(-2*log(S)/S);
}



/**
 * Matrix multiplication
 * c = a*b, where a is m-by-n and b is n-by-p
 * @param c c
 * @param a a
 * @param b b
 * @param m m dimension
 * @param n n dimension
 * @param p p dimension
 */
 inline void mult(double *c, const double *a, const double *b, int m, int n, int p, bool rm = true)
{
  memset(c, 0, m*p*sizeof(double));
  
  if (rm)
    for (int i=0; i<m; ++i) {
      for (int k=0; k<p; ++k) {
        double& cr = c[i*p+k];
        for (int j=0; j<n; ++j)
          cr += a[i*n+j]*b[j*p+k];
      }
    }
  else
    for (int i=0; i<m; ++i) {
      for (int k=0; k<p; ++k) {
        double& cr = c[i+k*m];
        for (int j=0; j<n; ++j)
          cr += a[i+j*m]*b[j+k*n];       
      }
    }
}
 
/**
 * Multiplication of a sparse matrix a and a regular matrix b
 * c = a*b, where a is m-by-n and b is n-by-p
 * @param c c
 * @param a a (an array of size an)
 * @param an number of nonzero elements in a
 * @param ai row indices of nonzero elements
 * @param aj column indices of nonzero elements
 * @param b b (an array of size n*p)
 * @param m m dimension
 * @param n n dimension
 * @param p p dimension
 */
void mult(double *c, 
          const double *a, int an, const int *ai, const int *aj, 
          const double *b, 
          int m, int n, int p);

/**
 * distance b/n two angles
 * @param a1 first angle
 * @param a2 second angle
 * @return distance b/n angles
 */
double dangle(double a1, double a2);

/**
 * load a PPM file and convert it into a character array
 * @param map the map
 * @param width map width
 * @param height map height
 * @param filename name of the ppm file
 */
void save_map(const char* map, int width, int height, const char* filename);

/**
 * save a character array map into a PPM file
 * @return the resulting map
 * @param width resulting width
 * @param height resulting height
 * @param filename name of the ppm file
 */
char* load_map(int* width, int* height, const char* filename);


void rot(double m[9], const double o[3]);

/**
 * 2-D Hermite interpolation between two vectors
 * the second vector is computed as the direction b/n xi and xf
 * @param c1 resulting t-multiple coefficents
 * @param c2 resulting t^2-multiple coefficents
 * @param c3 resulting t^3-multiple coefficents
 * @param xi initial position
 * @param vi initial velocity
 * @param xf final position
 */
void hermite2(double c1[2], double c2[2], double c3[2], 
              const double xi[2], const double vi[2],
              const double xf[2]);


/**
 * 3-D Hermite interpolation between two vectors
 * the second vector is computed as the direction b/n xi and xf
 * @param c1 resulting t-multiple coefficents
 * @param c2 resulting t^2-multiple coefficents
 * @param c3 resulting t^3-multiple coefficents
 * @param xi initial position
 * @param vi final velocity
 * @param xf final position
 */
void hermite3(double c1[3], double c2[3], double c3[3], 
              const double xi[3], const double vi[3],
              const double xf[3]);



/**
 * Normal distribution likelihood
 * @param mu mean
 * @param sigma variance
 * @param x sample
 * @return likelihood
 */
 double normal(double mu, double sigma, double x);

/**
 * Multivariate normal distribution likelihood with diagonal stdev
 * @param d dimension
 * @param mu mean
 * @param sigma variance
 * @param x sample
 * @return likelihood
 */
 double normal(int d, const double *mu, const double *sigma, const double *x);

/**
 * Multivariate normal distribution likelihood with diagonal stdev
 * @param d dimension
 * @param mu mean
 * @param sigma variance
 * @param x sample
 * @return likelihood
 */
 double normal2(const double mu[2], const double sigma[2], const double x[2]);

/**
 * 0-mean unit-stdev normal distribution 1-D sample
 * @return sample
 */
 double randn();

/**
 * Normal CDF a.k.a Phi(x)
 * @param x the argument
 * @return the CDF value
 */
double ncdf(double x);

}

#endif
