// This file is part of libgcop, a library for Geometric Control, Optimization, and Planning (GCOP)
//
// Copyright (C) 2004-2014 Marin Kobilarov <marin(at)jhu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef GCOP_NORMAL_H
#define GCOP_NORMAL_H

#include <Eigen/Dense>
#include <vector>
#include "utils.h"
#include <iostream>

namespace gcop {
  
  using namespace Eigen;
  using namespace std;

  template <int _n = Dynamic>
  class Normal {
  public:
  typedef Matrix<double, _n, 1> Vectornd;
  typedef Matrix<double, _n, _n> Matrixnd;

  /**
   * n-dimensional normal distribution
   * @param n dimension
   */
  Normal(int n = 1);
  
  /**
   * n-dimensional normal distribution with mean mu and covariance P
   * @param mu mean
   * @param P covariance
   */    
  Normal(const Vectornd &mu, const Matrixnd &P);
  
  virtual ~Normal();
  
  /**
   * Compute likelihood of element x
   * @param x n-dimensional vector
   * @return likelihood of sample
   */
  double L(const Vectornd &x) const;

  /**
   * Compute log likelihood of element x
   * @param x n-dimensional vector
   * @return likelihood of sample
   */
  double logL(const Vectornd &x) const;
  
  
  /**
   * Sample from the distribution
   * @param x n-dimensional vector to be sampled
   * @return likelihood of sample
   */
  double Sample(Vectornd &x);    
  
  /**
   * Updates the Cholesky factor and the normalization constant
   * @return true if covariance is positive definite
   */
  bool Update();
  
  /**
   * Estimate the distribution using data xs and costs cs (optional)
   * @param xps data points and corresponding probabilities (should sum up to 1)
   * @param a smoothing parameter [ mu_new = a*mu + (1-a)*mu_old ], equation to 1 by default
   */
  void Fit(const vector<pair<Vectornd, double> > xps, double a = 1);

  void Print(std::ostream &os) const;

  template<int _m>
  friend std::ostream& operator<<(std::ostream &os, const Normal<_m> n);
  
  Vectornd mu;     ///< mean
  Matrixnd P;      ///< covariance
  
  double det;      ///< determinant
  Matrixnd Pinv;   ///< covariance inverse
  bool pd;         ///< covariance is positive-definite
  
  Matrixnd A;      ///< cholesky factor
  Vectornd rn;     ///< normal random vector
  
  double norm;     ///< normalizer
  
  int bd;          ///< force a block-diagonal structure with block dimension bd (0 by default means do not enforce)
  
  bool bounded;    ///< whether to enforce a box support (false by default)
  Vectornd lb;     ///< lower bound
  Vectornd ub;     ///< upper bound
  
  LLT<Matrixnd> llt; ///< LLT object to Cholesky
  };
  


  template<int _n>
    Normal<_n>::Normal(int n):
    det(0),
    pd(false),
    norm(0),
    bd(0), 
    bounded(false) {
    
    if (_n == Dynamic) {
      mu.resize(n);
      P.resize(n,n);
      Pinv.resize(n,n);
      A.resize(n,n);
      rn.resize(n);
      lb.resize(n);
      ub.resize(n);
    }
    mu.setZero();
    P.setZero();
    Pinv.setZero();
    A.setZero();
    rn.setZero();
  }
  
  template<int _n>
    Normal<_n>::Normal(const Vectornd &mu, const Matrixnd &P):
    mu(mu),
    P(P),
    det(0),
    pd(false),
    norm(0),
    bd(0),
    bounded(false) {
    
    int n = mu.size();

    if (_n == Dynamic) {
      Pinv.resize(n,n);
      A.resize(n,n);
      rn.resize(n);
      lb.resize(n);
      ub.resize(n);
    }
    Pinv.setZero();
    A.setZero();
    rn.setZero();
    
    
    Update();
  }
  

  template<int _n>
    Normal<_n>::~Normal()
    {
    }

  template<int _n>
    double Normal<_n>::logL(const Vectornd &x) const
    {
      //if (!pd) {
      //  cout << "[W] Normal::L: not positive definite!" << endl;
      //}
      
      Vectornd d = x - mu;
      return -(d.dot(Pinv*d)/2) - (norm);
    }

  template<int _n> inline
    double Normal<_n>::L(const Vectornd &x) const
    {
      //if (!pd) {
      //  cout << "[W] Normal::L: not positive definite!" << endl;
      //}
      
      Vectornd d = x - mu;
      return exp(-d.dot(Pinv*d)/2)/norm;
    }
  
  template<int _n>
    bool Normal<_n>::Update()
    {
      llt.compute(P);
      
      if (llt.info() == Eigen::Success) {
        A = llt.matrixL();
        Pinv = P.inverse();
        //std::cout << "P INV = \n" << Pinv << "\n\n";
        det = P.determinant();
        //norm = sqrt(det); //sqrt(pow(2*M_PI, mu.size())*det);
        norm = sqrt(pow(2*M_PI, mu.size())*det);
        pd = true;
      } else {
        cout << "[W] Normal::Update: cholesky failed: P=\n" << P << endl;
        cout << "mu=\n" << mu.transpose() << endl;
        pd = false;    
      }
      
      return pd;
    }

  template<int _n>
    double Normal<_n>::Sample(Vectornd &x)
    {
      double p = 1;
      for (int i = 0; i < rn.size(); ++i) {
        rn(i) = random_normal();
        p *= rn(i);
      }
      
      x = mu + A*rn;
      
      if (bounded) {
        x = x.cwiseMax(lb);
        x = x.cwiseMin(ub);
      }

      return p;
    }
  
  template<int _n>
    void Normal<_n>::Fit(const vector<pair<Vectornd, double> > xws, double a)
    {
      int N = xws.size();
      assert(false);

      // sanity check
      {
        double weight_sum;
        for (auto &pair: xws) {
          weight_sum += pair.second;
        }
        //std::cout << "weights added up to " << weight_sum << std::endl;
        assert(fabs(weight_sum - 1) < 1e-5);
      }
      
      Vectornd mu;
      if (_n == Dynamic)
        mu.resize(this->mu.size());

      mu.setZero();
      for (int j = 0; j < N; ++j) {
        const pair<Vectornd, double> &xw = xws[j];
        mu += xw.first*xw.second;
      }
      
      Matrixnd P;
      if (_n == Dynamic)
        P.resize(this->mu.size(), this->mu.size());
      P.setZero();

      for (int j = 0; j < N; ++j) {
        const pair<Vectornd, double> &xw = xws[j];
        Vectornd dx = xw.first - mu;
        if (!bd) {
          P += (xw.second*dx)*dx.transpose();
        } else {
          int b = dx.size()/bd;
          for (int i = 0; i < b; ++i) {
            int bi = i*bd;
            Vectornd bdx = dx.segment(bi, bd);        
            P.block(bi, bi, bd, bd) += (xw.second*bdx)*bdx.transpose();       
          }
        }
      }
      
      if (fabs(a-1) < 1e-16) {
        this->mu = mu;
        this->P = P;
      } else {
        this->mu = a*mu + (1-a)*this->mu;
        this->P = a*P + (1-a)*this->P;
      }
    }  

  template<int _n>
  void Normal<_n>::Print(std::ostream &os) const {
    os <<"mu=";
    for (int i = 0; i < mu.size(); ++i) {
      os << mu(i) << ",";
    }
    os << std::endl;
    std::cout <<"P=\n"<<P<<"\n";
  }

  template<int _n>
  std::ostream& operator<<(std::ostream &os, const Normal<_n> n) {
    n.Print(os);
    return os;
  }

}


#endif
