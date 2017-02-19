// This file is part of libgcop, a library for Geometric Control, Optimization, and Planning (GCOP)
//
// Copyright (C) 2004-2014 Marin Kobilarov <marin(at)jhu.edu>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef GCOP_GMM_H
#define GCOP_GMM_H

#include "normal.h"
#include <iostream>

//#define DEBUG

namespace gcop {
  using namespace Eigen;
  using namespace std;

  /**
   * Gaussian mixture model
   *
   * Author: Marin Kobilarov marin(at)jhu.edu
   */
  template<int _n = Dynamic>
    class Gmm {
    public:
      typedef Matrix<double, _n, 1> Vectornd;
      typedef Matrix<double, _n, _n> Matrixnd;

      /**
       * Construct a GMM with dimension n and k clusters
       * @param n dimension
       * @arapm k number of modes
       */
      Gmm(int n, int k = 1);

      virtual ~Gmm();

      bool Update();

      double L(const Vectornd &x) const;

      double logL(const Vectornd &x) const;

      double Sample(Vectornd &x);

      void Print(std::ostream &os) const;

  template<int _m>
      friend std::ostream& operator<<(std::ostream &os, const Gmm<_m> g);

      void Init(const Vectornd &xlb, const Vectornd &xub);

      /**
       * Fit a GMM to data
       * @param xps data (pairs of vectors and weights), weights should add to one
       * @param a smoothing factor for updating the parameter v according to
       *         [ v_new = a*v_est + (1-a)*v_old ]; it set to 1 by default
       * @param iter maximum number of EM iterations (only applies for multiple mixtures)
       * @param S use additional noise matrix during EM for stability
       */
      void Fit(const vector<pair<Vectornd, double> > &xps, double a = 1,
               int iter = 50, const Matrixnd *S = 0);

      int k;              ///< number of mixture components
      vector<Normal<_n>> ns;  ///< normal distribution for each component   
      vector<double> ws;  ///< component weights
      vector<double> cdf; ///< CDF

      double tol; 

    protected:
      Vectornd t2;
      Matrixnd t3;
    };


  template<int _n>
    Gmm<_n>::Gmm(int n, int k) :
      k(k), ns(k, Normal<_n>(n)), ws(k), cdf(k), tol(0.01)
  {
    assert(n > 0);
    assert(k > 0);

    for (int i = 0; i < k; ++i) {
      //    ns[i] = new Normal(n);
      ws[i] = 1/((double)k);
      cdf[i] = (i+1)/((double)k);

      ns[i].mu.setZero();
      ns[i].P = VectorXd::Constant(n, 1).asDiagonal();
    }

    if (_n == Dynamic) {
      t2.resize(n);
      t3.resize(n, n);    
    }
  }


  template<int _n>
    Gmm<_n>::~Gmm()
    {
    }


  template<int _n>
    bool Gmm<_n>::Update()
    {
      if (k == 1) {
        ws[0] = 1;
        cdf[0] = 1;
        return ns[0].Update();

      } else {
        /*
           bool ok = true;
           double wn = 0;
           for (int i = 0; i < k; ++i) {
           ok = ok && ns[i]->Update();
           wn += ws[i];
           }

           for (int i = 0; i < k; ++i) {
           ws[i] /= wn;
           cdf[i] = (i ? cdf[i - 1] + ws[i] : ws[i]); 
           }
           return ok;
           */    
        // No need to call update since it is called after EM

        bool ok = true;
        double wt_norm = 0;
        for (int i = 0; i < k; ++i) {
          wt_norm += ws[i];
        }
        for (int i = 0; i < k; ++i) {
          ws[i] /= wt_norm;
          cdf[i] = (i ? cdf[i - 1] + ws[i] : ws[i]); 
          ok = ok && ns[i].pd;
        }
        return ok;
      }
    }

  template<int _n>
    double Gmm<_n>::L(const Vectornd &x) const
    {
      if (k == 1) {
        return ns[0].L(x);
      } else {

        double l = 0;
        for (int i = 0; i < k; ++i)
          l += ws[i]*ns[i].L(x);
        return l;
      }
    }
  template<int _n> inline
    double Gmm<_n>::logL(const Vectornd &x) const
    {
      if (k == 1) {
        return ns[0].logL(x);
      } else {

        double l = 0;
        for (int i = 0; i < k; ++i)
          l += ws[i]*ns[i].L(x);    
        return log(l);
      }
    }

  template<int _n>
    double Gmm<_n>::Sample(Vectornd &x)
    {
      if (k == 1) {
        return ns[0].Sample(x);

      } else {
        // for now this is unefficient if k is big
        // TODO: implement as binary search
        double uc = rand()/(double)RAND_MAX;
        int i = 0;
        while (uc > cdf[i])
          ++i;

        assert(i < k);
        return ns[i].Sample(x);
      }
    }

  template<int _n>
    void Gmm<_n>::Init(const Vectornd &xlb, const Vectornd &xub)
    {
      Vectornd dx = xub - xlb;   // range
      Vectornd r = dx/pow(k, 1.0/dx.size());///2;     // radius
      Matrixnd P = (r.cwiseProduct(r)).asDiagonal();
      for (int i = 0; i < k; ++i) {    
        ns[i].mu = xlb + dx.cwiseProduct(VectorXd::Random(xlb.size()));
        ns[i].P = P;
        ws[i] = 1.0/k;
        ns[i].Update();
      }

      Update();
    }


  template<int _n>
    void Gmm<_n>::Fit(const vector<pair<Vectornd, double> > &xps, double a, int iter, const Matrixnd *S)
    {
      int N = xps.size();

      // sanity check
      {
        double weight_sum = 0;
        for (auto &pair: xps) {
          weight_sum += pair.second;
        }
        if (fabs(weight_sum - 1) >= tol) {
          //std::cout << "Weights added up to " << weight_sum << "... " << fabs(weight_sum - 1) << std::endl;
          assert(fabs(weight_sum - 1) < tol);
        }
      }

      if (false and k == 1) {
        ns[0].Fit(xps, a);
        if (S)
          ns[0].P += *S;
        return;
      }

      assert(N > 0);

      double ps[N][k];

      for (int l = 0; l < iter; ++l) {

        // E-step

        for (int j = 0; j < N; ++j) {

          const Vectornd &x = xps[j].first;      
          double p = xps[j].second;

          double norm = 0;
          double *psj = ps[j];

          for (int i = 0; i < k; ++i) {
            psj[i] = p*ns[i].L(x);    // likelihood of each sample        
            norm += psj[i];
          }

          if (not norm == 0) {
#ifdef DEBUG
          //assert(norm > 1e-10);
          cout << norm << ":";
          cout << "    normalized: ps[" << j << "]=";
#endif
          for (int i = 0; i < k; ++i) {
            psj[i] /= norm;
#ifdef DEBUG
            cout << psj[i] << " ";
#endif
          }    
#ifdef DEBUG
          cout << endl;
#endif
          } else {
            for (int i = 0; i < k; ++i) {
              psj[i] = 1.0 / k;
            }
          }
        }  



        // M-step
        double maxd = 0;

        for (int i = 0; i < k; ++i) {
          double t1 = 0;
          //      Vectornd t2; = VectorXd::Zero(ns[0].mu.size());
          t2.setZero();//Redundancy
          //MatrixXd t3 = MatrixXd::Zero(ns[0].mu.size(), ns[0].mu.size());
          t3.setZero();

          for (int j = 0; j < N; ++j) {
            const VectorXd &x = xps[j].first;
            t1 += ps[j][i];
            t2 += ps[j][i]*x;
            t3 += (ps[j][i]*x)*x.transpose();
          }

          ws[i] = t1/N;

          VectorXd mu = t2/t1;

          double d = (mu - ns[i].mu).norm();
          if (maxd < d)
            maxd = d;

          ns[i].mu = mu;
          //ns[i].P = (t3 - t2*(t2.transpose()/t1))/t1;        

#if 1
          ns[i].P.setZero();
          for (int j = 0; j < N; ++j) {
            const VectorXd &x = xps[j].first;
            VectorXd Px = ((ps[j][i]*x) - mu);
            ns[i].P += Px * Px.transpose() / t1;
          }
#endif
          if (S)
            ns[i].P += *S;

          //std::cout << "i=" << i << ":" << ns[i].mu.transpose() << "\n";

          if (!ns[i].Update()) // set Pinv, det, norm
            return;
        }

        if (maxd < tol) {
          Update();
          //cout << "[W] Gmm::Fit: tolerance " << maxd << " reached after " << l << " iterations!" << endl;
          break;
        }
      }
    }

  template<int _n>
    void Gmm<_n>::Print(std::ostream &os) const {
      os << "k=" << k << std::endl;
      for(int i = 0; i < k; ++i) {
        os << "weight=" << ws[i] << std::endl;
        ns[i].Print(os);
      }
    }

  template<int _n>
  std::ostream& operator<<(std::ostream &os, const Gmm<_n> n) {
    n.Print(os);
    return os;
  }

}

#endif
