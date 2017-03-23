// This file is part of libgcop, a library for Geometric Control, Optimization, and Planning (GCOP)
//
// Copyright (C) 2004-2014 Marin Kobilarov <marin(at)jhu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <costar_task_plan/dist/utils.h>
#include <assert.h>
#include <iostream>

using namespace std;
using namespace gcop;

namespace gcop {

void mult(double *c, 
          const double *a, int an, const int *ai, const int *aj, 
          const double *b, 
          int m, int n, int p)
{
  
  memset(c, 0, m*p*sizeof(double));
  int ao = 0;
  for (int i=0; i<m; ++i) {
    int j = 0;
    for (int k=0; k<p; ++k) {
      double& cr = c[i*p+k];
      for (j=0; ao+j < an && (ai[ao+j]-1)==i; ++j) {
        cr += a[ao+j]*b[(aj[ao+j]-1)*p+k];
      }
    }
    ao += j;
  }
  assert(ao==an);
}


void save_map(const char* map, int width, int height, const char* filename)
{
  int i, ind;
  char data[width*height*3];
  FILE* file = fopen(filename, "w");
  assert(file);
  fprintf(file, "P6\n%d %d 255\n", width, height);
  ind = 0;
  for (i = 0; i < width*height; i++, ind+=3) {
    data[ind] = data[ind+1] = data[ind+2] = (char)(map[i]*100);
  }
  assert(ind == 3*width*height);
  assert((int)fwrite(data, sizeof(char), ind, file) == ind);
  fclose(file);
}

char* load_map(int* width, int* height, const char* filename)
{
  int i, size;
  char *map, *data;
  FILE* file = fopen(filename, "r");
  assert(file);
  int n = fscanf(file, "P6\n%d %d 255\n", width, height);
  size = (*width**height);
  map = (char*)malloc(size);
  data = (char*)malloc(size*3);
  n = fread(data, sizeof(char), size*3, file);
  for (i = 0; i < size; i++)
    map[i] = (data[3*i] ? 1 : 0);
  free(data);
  fclose(file);
  return map;
}


void rot(double m[9], const double o[3])
{
  double ca = cos(o[0]);
  double sa = sin(o[0]);
  double cb = cos(o[1]);
  double sb = sin(o[1]);
  double cc = cos(o[2]);
  double sc = sin(o[2]);
  m[0] = ca*cb;
  m[1] = ca*sb*sc-sa*cc;
  m[2] = ca*sb*cc+sa*sc;
  m[3] = sa*cb;
  m[4] = sa*sb*sc+ca*cc;
  m[5] = sa*sb*cc-ca*sc;
  m[6] = -sb;
  m[7] = cb*sc;
  m[8] = cb*cc;
}



double dangle(double a1, double a2)
{
  double c1 = cos(a1);
  double s1 = sin(a1);
  double c2 = cos(a2);
  double s2 = sin(a2);
  //  return atan2(s2-s1,c2-c1);
  return sqrt((c1-c2)*(c1-c2)+(s1-s2)*(s1-s2));
}


void hermite2(double c1[2], double c2[2], double c3[2], 
              const double xi[2], const double vi[2],
              const double xf[2])
{
  double d[2];
  MINUS(d, xf, xi);
  double vn = NORM(vi);
  double dn = NORM(d);
  MULT(c1, d, vn/dn);
  c2[0] = 3*(xf[0]-xi[0])-c1[0]-2*vi[0];
  c2[1] = 3*(xf[1]-xi[1])-c1[1]-2*vi[1];
  c3[0] = -2*(xf[0]-xi[0]) + c1[0] + vi[0];
  c3[1] = -2*(xf[1]-xi[1]) + c1[1] + vi[1];
}

void hermite3(double c1[3], double c2[3], double c3[3], 
              const double xi[3], const double vi[3],
              const double xf[3])
{
  double d[3];
  MINUS3(d, xf, xi);
  double vn = NORM3(vi);
  double dn = NORM3(d);
  MULT(c1, d, vn/dn);
  c2[0] = 3*(xf[0]-xi[0])-c1[0]-2*vi[0];
  c2[1] = 3*(xf[1]-xi[1])-c1[1]-2*vi[1];
  c2[2] = 3*(xf[2]-xi[2])-c1[2]-2*vi[2];
  c3[0] = -2*(xf[0]-xi[0]) + c1[0] + vi[0];
  c3[1] = -2*(xf[1]-xi[1]) + c1[1] + vi[1];
  c3[2] = -2*(xf[2]-xi[2]) + c1[2] + vi[2];
}


/**
 * Normal distribution likelihood
 * @param mu mean
 * @param sigma variance
 * @param x sample
 * @return likelihood
 */
double normal(double mu, double sigma, double x) 
{
  return exp(-(x - mu)*(x - mu)/(2*sigma*sigma))/(sigma*sqrt(2*M_PI));
}

/**
 * Multivariate normal distribution likelihood with diagonal stdev
 * @param d dimension
 * @param mu mean
 * @param sigma variance
 * @param x sample
 * @return likelihood
 */
double normal(int d, const double *mu, const double *sigma, const double *x) 
{
  double v = 0;
  for (int i = 0; i < d; ++i)
    v = v -(x[i] - mu[i])*(x[i] - mu[i])/(2*sigma[i]*sigma[i]);
  return exp(v);
}

/**
 * Multivariate normal distribution likelihood with diagonal stdev
 * @param d dimension
 * @param mu mean
 * @param sigma variance
 * @param x sample
 * @return likelihood
 */
double normal2(const double mu[2], const double sigma[2], const double x[2]) 
{
  return exp(-(x[0] - mu[0])*(x[0] - mu[0])/(2*sigma[0]*sigma[0]) - (x[1] - mu[1])*(x[1] - mu[1])/(2*sigma[1]*sigma[1]));
}


/**
 * 0-mean unit-stdev normal distribution 1-D sample
 * @return sample
 */
double randn()
{
  static const double r_max=RAND_MAX;
  double U1,U2,V1,V2;
  double S = 2;
  while(S>=1) {
    U1 = rand()/r_max;
    U2 = rand()/r_max;
    V1 = 2*U1-1;
    V2 = 2*U2-1;
    S = V1*V1+V2*V2;
  }
  return V1*sqrt(-2*log(S)/S);
}

double ncdf(double x)
{
    // constants
  static const double a1 =  0.254829592;
  static const double a2 = -0.284496736;
  static const double a3 =  1.421413741;
  static const double a4 = -1.453152027;
  static const double a5 =  1.061405429;
  static const double p  =  0.3275911;
  
  // Save the sign of x
  int sign = 1;
  if (x < 0)
    sign = -1;
  x = fabs(x)/sqrt(2.0);
  
  // A&S formula 7.1.26
  double t = 1.0/(1.0 + p*x);
  double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
  
  return 0.5*(1.0 + sign*y);
}


};
