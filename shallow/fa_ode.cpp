/*
 * File:    fa_ode.cpp
 *
 * Author: Maria Refinetti <mariaref@gmail.com>
 *         Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Version: 0.1
 *
 * Date:    August 2020
 */

#include <cmath>
#include <getopt.h>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

#include <armadillo>
#include <chrono>

#include "libscmpp.h"

using namespace std;
using namespace arma;

const int NUM_LOGPOINTS = 200;  // # of times we print something

const char * usage = R"USAGE(

This is fa++, a tool to analyse training by (direct) feedback alignment in
two-layer neural networks in the teacher-student setup. This code implements the analytical ODEs and integrates them.

usage: scmpp.exe [-h] [--g G] [-N N] [-M M] [-K K] [--lr LR]
                     [--init INIT] [--steps STEPS] [--uniform A] [--sigma SIGMA]
                     [-s SEED] [--quiet]

optional arguments:
  -h, -?                show this help message and exit
  --g G               activation function for the teacher;
                           0-> linear, 1->erf, 2->ReLU.
  -N, --N N             input dimension
  -M, --M M             number of hidden units in the teacher network
  -K, --K K             number of hidden units in the student network
  -l, --lr LR           learning rate
  -a, --steps STEPS     max. weight update steps in multiples of N
  --fa                  train with feedback alignment
  --uniform A           make all of the teacher's second layer weights equal to
                          this value. If the second layer of the student is not
                          trained, the second-layer output weights of the student
                          are also set to this value.
  -s, --sigma SIGMA     std. dev. of teacher's output noise. Default=0.
                        For classification, the probability that a label is
                        drawn at random.
  -i, --init INIT       weight initialisation:
                          1,2: i.i.d. Gaussians with sd 1 or 1/sqrt(N), resp.
                            3: informed initialisation; only for K \ge M.
                            4: denoising initialisation
                            5: 'mixed': i.i.d. Gaussian with 1/sqrt(N), 1/sqrt(K)
  --stop                generalisation error at which to stop the simulation.
  --store               store initial overlap and final weight matrices.
  -r SEED, --seed SEED  random number generator seed. Default=0
  --dummy               dummy command that doesn't do anything but helps with parallel.
  --quiet               be quiet and don't print order parameters to cout.
)USAGE";


/**
 * Returns the projection of the given covariance matrix C to the d.o.f. a, b.
 */
void update_C2(mat& C2, mat& cov, int a, int b) {
  // The code below is a brute-force implementation of the following code:
  // mat A = mat(size(cov), fill::zeros);
  // A(0, a) = 1;
  // A(1, b) = 1;
  // return A * cov * A.t();
  
  C2(0, 0) = cov(a, a);
  C2(0, 1) = cov(a, b);
  C2(1, 0) = cov(b, a);
  C2(1, 1) = cov(b, b);
}

double J2_lin(mat& C) {
  return 1;
}

double J2_erf(mat& C) {
  return 2 / datum::pi /
      sqrt(1 + C(0, 0) + C(1, 1) - pow(C(0, 1), 2) + C(0, 0) * C(1, 1));
}

double I2_erf(mat& C) {
  return (2. / datum::pi * asin(C(0, 1)/(sqrt(1 + C(0, 0))*sqrt(1 + C(1, 1)))));
}

double I2_lin(mat& C) {
  return C(0, 1);
}

/**
 * Returns the projection of the given covariance matrix C to the d.o.f. a, b,
 * and c.
 */
void update_C3(mat& C3, mat& cov, int a, int b, int c) {
  // The code below is a brute-force implementation of the following code:
  // mat A = mat(size(cov), fill::zeros);
  // A(0, a) = 1;
  // A(1, b) = 1;
  // A(2, c) = 1;
  // return A * cov * A.t();
  
  C3(0, 0) = cov(a, a);
  C3(0, 1) = cov(a, b);
  C3(0, 2) = cov(a, c);
  C3(1, 0) = cov(b, a);
  C3(1, 1) = cov(b, b);
  C3(1, 2) = cov(b, c);
  C3(2, 0) = cov(c, a);
  C3(2, 1) = cov(c, b);
  C3(2, 2) = cov(c, c);
}

double I3_erf(mat& C) {
  double lambda3 = (1 + C(0, 0))*(1 + C(2, 2)) - pow(C(0, 2), 2);

  return (2. / datum::pi / sqrt(lambda3) *
          (C(1, 2)*(1 + C(0, 0)) - C(0, 1)*C(0, 2)) / (1 + C(0, 0)));
}

double I3_lin(mat& C) {
  return C(1, 2);
}


/**
 * Returns the projection of the given covariance matrix C to the d.o.f. a, b,
 * c, and d.
 */
void update_C4(mat& C4, mat& cov, int a, int b, int c, int d) {
  // The code below is a brute-force implementation of the following code:
  // mat A = mat(size(cov), fill::zeros);
  // A(0, a) = 1;
  // A(1, b) = 1;
  // A(2, c) = 1;
  // A(3, d) = 1;
  // return A * cov * A.t();

  C4(0, 0) = cov(a, a);
  C4(0, 1) = cov(a, b);
  C4(0, 2) = cov(a, c);
  C4(0, 3) = cov(a, d);
  C4(1, 0) = cov(b, a);
  C4(1, 1) = cov(b, b);
  C4(1, 2) = cov(b, c);
  C4(1, 3) = cov(b, d);
  C4(2, 0) = cov(c, a);
  C4(2, 1) = cov(c, b);
  C4(2, 2) = cov(c, c);
  C4(2, 3) = cov(c, d);
  C4(3, 0) = cov(d, a);
  C4(3, 1) = cov(d, b);
  C4(3, 2) = cov(d, c);
  C4(3, 3) = cov(d, d);
}

double I4_erf(mat& C) {
  double lambda4 = (1 + C(0, 0))*(1 + C(1, 1)) - pow(C(0, 1), 2);

  double lambda0 = (lambda4 * C(2, 3)
                    - C(1, 2) * C(1, 3) * (1 + C(0, 0))
                    - C(0, 2)*C(0, 3)*(1 + C(1, 1))
                    + C(0, 1)*C(0, 2)*C(1, 3)
                    + C(0, 1)*C(0, 3)*C(1, 2));
  double lambda1 = (lambda4 * (1 + C(2, 2))
                    - pow(C(1, 2), 2) * (1 + C(0, 0))
                    - pow(C(0, 2), 2) * (1 + C(1, 1))
                    + 2 * C(0, 1) * C(0, 2) * C(1, 2));
  double lambda2 = (lambda4 * (1 + C(3, 3))
                    - pow(C(1, 3), 2) * (1 + C(0, 0))
                    - pow(C(0, 3), 2) * (1 + C(1, 1))
                    + 2 * C(0, 1) * C(0, 3) * C(1, 3));

  return (4 / pow(datum::pi, 2) / sqrt(lambda4) *
          asin(lambda0 / sqrt(lambda1 * lambda2)));
}

double I4_lin(mat& C) {
  return C(2, 3);
}


/**
 * Performs an integration step and returns increments for Q and R.
 * Parameters:
 * -----------
 * duration:
 *     the time interval for which to propagate the system
 * dt :
 *     the length of a single integration step
 * t :
 *     time at the start of the propagation
 * Q : (K, K)
 *     student-student overlap
 * R : (K, M)
 *     student-teacher overlap
 * T : (M, M)
 *     teacher-teacher overlap
 * A : vec (M)
 *     hidden unit-to-output weights of the teacher
 * v : (K)
 *     hidden unit-to-output weights of the student
 * lr : scalar
 *     learning rate of the first layer
 * lr2 : scalar
 *     learning rate of the second layer
 * wd : scalar
 *     weight decay constant
 * sigma : double
 *     std. dev. of the teacher's output noise
*/
void propagate(double duration, double dt, double& time,
               mat& Q, mat& R, mat& T, vec& A, vec& v, vec& v_rand, double(*J2)(mat&),
               double(*I2)(mat&), double(*I3)(mat&), double(*I4)(mat&),
               double lr,double sigma, bool both) {
  int K = R.n_rows;
  int M = R.n_cols;
  cout<<" sigma is "<<sigma<<endl;

  double propagation_time = 0;
  // construct the covariance matrix C
  mat C = zeros(K + M, K + M);
  mat C2 = zeros(2, 2);
  mat C3 = zeros(3, 3);
  mat C4 = zeros(4, 4);
  while(propagation_time < duration) {
    // update the full covariance matrix of all local fields
    C.submat(0, 0, K-1, K-1) = Q;
    C.submat(0, K, K-1, K+M-1) = R;
    C.submat(K, 0, K+M-1, K-1) = R.t();
    C.submat(K, K, K+M-1, K+M-1) = T;

    // integrate R
    for (int i = 0; i < K; i++) { // student
      for (int n = 0; n < M; n++) { // teacher
        // weight decay
//        R(i, n) -= dt * wd * R(i, n);

        for (int m = 0; m < M; m++) { // teacher
          update_C3(C3, C, i, K+n, K+m);
          R(i, n) += dt * lr * v_rand(i) * A(m) * I3(C3);
        }
        for (int j  = 0; j < K; j++) {  // student
          update_C3(C3, C, i, K+n, j);
          R(i, n) -= dt * lr * v_rand(i) * v(j) * I3(C3);
        }
      }
    }

    // integrate Q
    for (int i = 0; i < K; i++) {  // student
      for (int k = i; k < K; k++) {  // student
        // weight decay
//        Q(i, k) -= dt * 2 * wd * Q(i, k);

        // terms proportional to the learning rate
        for (int m = 0; m < M; m++){ // teacher
          update_C3(C3, C, i, k, K + m);
          Q(i, k) += dt * lr * v_rand(i) * A(m) * I3(C3);
          update_C3(C3, C, k, i, K + m);
          Q(i, k) += dt * lr * v_rand(k) * A(m) * I3(C3);
        }
        for (int j = 0; j < K; j++) { // student
          update_C3(C3, C, i, k, j);
          Q(i, k) -= dt * lr * v_rand(i) * v(j) * I3(C3);
          update_C3(C3, C, k, i, j);
          Q(i, k) -= dt * lr * v_rand(k) * v(j) * I3(C3);
        }
          
        // noise term
        if (sigma > 0) {
          //Q(i, k) += dt * v(i) * v(k) * (pow(lr, 2) * pow(sigma, 2) * 2 / datum::pi /
          //                 sqrt(1+Q(i, i)+Q(k, k)-pow(Q(i, k), 2)+Q(i, i)*Q(k, k)));
          // Q(i, k) += dt * v(i) * v(k) * (pow(lr, 2) * pow(sigma, 2));
          update_C2(C2, C, i, k);
          Q(i, k) += dt * v_rand(i) * v_rand(k) * (pow(lr, 2) * pow(sigma, 2)) * J2(C2);
        }
        
        // SGD terms quadratic to the learning rate squared
        for (int n = 0; n < M; n++) {  // teacher
          for (int m = 0; m < M; m++) {  // teacher
            update_C4(C4, C, i, k, K + n, K + m);
            Q(i, k) += dt * pow(lr, 2) * v_rand(i) * v_rand(k) * A(n) * A(m) * I4(C4);
          }
        }

        for (int j = 0; j < K; j++) {  // student
          for (int n = 0; n < M; n++) {  // teacher
            update_C4(C4, C, i, k, j, K + n);
            Q(i, k) -= dt * pow(lr, 2) * v_rand(i) * v_rand(k) * v(j) * A(n) * 2 * I4(C4);
          }
        }

        for (int j = 0; j < K; j++) {  // student
          for (int l = 0; l < K; l++) {  // student
            update_C4(C4, C, i, k, j, l);
            Q(i, k) += dt * pow(lr, 2) * v_rand(i) * v_rand(k) * v(j) * v(l) * I4(C4);
          }
        }
      }
    }
    Q = symmatu(Q); // copy the upper half of the matrix to its lower part
    
    // integrate v
    vec v_new = vec(size(v), fill::zeros);
    if (both) {
      for (int i = 0; i < K; i++) {  // student
        // weight decay (?)
//        v_new(i) -= dt * wd * v(i);

        for (int k = 0; k < K; k++) {  // student
          update_C2(C2, C, i, k);
          v_new(i) -= dt * lr * v(k) * I2(C2);
        }

        for (int n = 0; n < M; n++) { // teacher
          update_C2(C2, C, i, K + n);
          v_new(i) += dt * lr * A(n) * I2(C2);
        }

        // no terms due to teacher's output noise
      }
    }

    if (both)
      v += v_new;
    time += dt;
    propagation_time += dt;
  }
}


int main(int argc, char* argv[]) {
   // flags; false=0 and true=1
    int both  = 1;      // train both layers
    int fa    = 1;      // train using feedback alignment
    int quiet = 0;  // don't print the order parameters to cout
    // other parameters
    int    g         = ERF;   // teacher and student activation function
    int    N         = 1000;  // number of eigenvalues
    int    M         = 2;  // num of teacher's hidden units
    int    K         = 2;  // num of student's hidden units
    double lr        = 0.2;  // learning rate
    double sigma     = 0;  // std.dev. of the teacher's additive output noise
    double dt        = 0.01;
    int    init      = INIT_LARGE; // initialisation
    string prefix;  // file name prefix to preload the weights
    double max_steps = 1000;  // max number of gradient updates / N
    int    seed      = 0;  // random number generator seed
    
    // parse command line options using getopt
    int c;

  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"fa",         no_argument, &fa,             1},
    {"both",       no_argument, &both,           1},
    {"quiet",      no_argument, &quiet,          1},
    {"N",       required_argument, 0, 'N'},
    {"g",       required_argument, 0, 'g'},
    {"M",       required_argument, 0, 'M'},
    {"K",       required_argument, 0, 'K'},
    {"delta",   required_argument, 0, 'z'},
    {"lr",      required_argument, 0, 'l'},
    {"sigma",   required_argument, 0, 's'},
    {"init",    required_argument, 0, 'i'},
    {"prefix",  required_argument, 0, 'p'},
    {"overlap", required_argument, 0, 'o'},
    {"steps",   required_argument, 0, 'a'},
    {"seed",    required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };
  
  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "g:N:M:K:l:s:d:i:f:o:a:r",
                    long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) {
      break;
    }

    switch (c) {
      case 0:
        break;
      case 'N':
        N = atoi(optarg);
        break;
      case 'g':
        g = atoi(optarg);
        break;
      case 'M':
        M = atoi(optarg);
        break;
      case 's':
        sigma = atof(optarg);
        break;
      case 'K':
        K = atoi(optarg);
        break;
      case 'l':
        lr = atof(optarg);
        break;
      case 'i':  // initialisation of the weights
        init = atoi(optarg);
        break;
      case 'p':  // initialisation of the weights
        prefix = string(optarg);
        break;
      case 'd':  // integration time-step
        dt = atof(optarg);
        break;
      case 'a':  // number of steps
        max_steps = atof(optarg);
        break;
      case 'r':
        seed = atoi(optarg);
        break;
      case '?':
        cout << usage << endl;
        return 0;
      default:
        abort ();
    }
  }
  

  // set the seed
  arma_rng::set_seed(seed);

  double (*J2_fun)(mat&);
  double (*I2_fun)(mat&);
  double (*I3_fun)(mat&);
  double (*I4_fun)(mat&);
  mat (*g_fun)(mat&);
  switch (g) {  // find the teacher's activation function
    case LINEAR:
      J2_fun = J2_lin;
      I2_fun = I2_lin;
      I3_fun = I3_lin;
      I4_fun = I4_lin;
      g_fun = g_lin;
      break;
    case ERF:
      J2_fun = J2_erf;
      I2_fun = I2_erf;
      I3_fun = I3_erf;
      I4_fun = I4_erf;
      g_fun = g_erf;
      break;
    default:
      cerr << "g has to be linear (g=" << LINEAR << ") or erf (g=" << ERF
           << ").\n will exit now!" << endl;
      return 1;
  }
  
  dt= 1.0/N;
  const char* g_name = activation_name(g);
  FILE* logfile;
  if (prefix.empty()) {
    char* log_fname;
    asprintf(&log_fname,
             "fa%d_ode_%s_%s_%sM%d_K%d_lr%g_sigma%g_i%d_steps%g_dt%g_s%d.dat",
             fa,g_name, g_name, (both ? "both_" : ""), M, K, lr,sigma, init, max_steps, dt, seed);
    
    logfile = fopen(log_fname, "w");
  }
  else{
    string log_fname = prefix;
    log_fname.append("_ode.dat");
    logfile = fopen(log_fname.c_str(), "w");
  }
  
  ostringstream welcome;
  welcome << "# This is fa++ ODE integrator for two-layer NN" << endl
          << "# g1=g2=" << g_name << ", M=" << M << ", K=" << K
          << ", steps/N=" << max_steps<< ", sigma=" << sigma << ", seed=" << seed << endl
  << "# lr=" << lr <<endl;
  if (both) {
    welcome << "# training both layers";
    welcome << endl;
  }
  
  if (!prefix.empty()) {
    welcome << "# took initial conditions from simulation " << prefix << endl;
  }
  welcome << "# steps / N, eg, et, diff" << endl;
  string welcome_string = welcome.str();
  cout << welcome_string;
  
  fprintf(logfile, "%s", welcome_string.c_str());

  // self-overlap of the student
  mat Q = mat(K, K);
  mat R = mat(K, M);
  mat T = mat(M, M, fill::eye);
  vec A = vec(M, fill::ones);
  vec v = vec(K);
  vec v_rand = vec(K);
  
  //Load initial conditions from file
  if (!prefix.empty()) {
    prefix.append("_Q0.dat");
    bool ok = Q.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_R0.dat");
    ok = ok && R.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_v0.dat");
    ok = ok && v.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_A0.dat");
    ok = ok && A.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_T0.dat");
    ok = ok && T.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_vrand.dat");
    ok = ok && v_rand.load(prefix);
    if (!ok) {
      cerr << "Error loading initial conditions from files, will exit !" << endl;
      return 1;
    } else {
      cout << "# Loaded all initial conditions successfully." << endl;
    }
  }else{
      
      cout << "Generating initial conditions" << endl;
      v_rand=randn<vec>(K);
      mat w = randn<mat>(K, N);
      mat B = randn<mat>(M, N);
      A= ones(M);
      v= ones(K);
      Q=1. / N*w*w.t();
      R=1. / N*w*B.t();
      T=1. / N*B*B.t();
      
    }
  
  // find printing times
  vec print_times = logspace<vec>(-1, log10(max_steps), NUM_LOGPOINTS);
  print_times(0) = 0;
  vec durations = diff(print_times);
  
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  double t = 0;
  bool converged = false;
  for (double& duration : durations) {
    double eg = eg_analytical(Q, R, T, A, v, g_fun, g_fun);
    std::ostringstream msg;
    double diff = datum::nan;
    msg << t << ", " << eg << ", " << datum::nan << ", " << diff << ", ";
    if (!quiet) {
      for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
          msg << R(k, m) << ", ";
        }
      }
      for (int k = 0; k < K; k++) {
        for (int l = k; l < K; l++) {
          msg << Q(k, l) << ", ";
        }
      }
      for (int m = 0; m < M; m++) {
        for (int n = m; n < M; n++) {
          msg << T(m, n) << ", ";
        }
      }
      for (int m = 0; m < M; m++) {
        msg << A(m) << ", ";
      }
      for (int k = 0; k < K; k++) {
        msg << v(k) << ", ";
      }
      for (int k = 0; k < K; k++) {
        msg << v_rand(k) << ", ";
      }
    }
    
    std::string msg_str = msg.str();
    msg_str = msg_str.substr(0, msg_str.length() - 2);
    cout << msg_str << endl;
    fprintf(logfile, "%s\n", msg_str.c_str());
    fflush(logfile);

    if (eg < 1e-14 && t > 100) {
      converged = true;
      break;
    } else {
      if(fa){
      propagate(duration, dt, t, Q, R, T, A, v,v_rand,
                J2_fun, I2_fun, I3_fun, I4_fun,
                lr,sigma,both);}
      else{
        propagate(duration, dt, t, Q, R, T, A, v,v,
        J2_fun, I2_fun, I3_fun, I4_fun,
        sigma,lr,both);
      }
    }
  }
  if (!converged) {
    double eg = eg_analytical(Q, R, T, A, v, g_fun, g_fun);
    std::ostringstream msg;
    double diff = datum::nan;
    msg << t << ", " << eg << ", " << datum::nan << ", " << diff << ", ";
    if (!quiet) {
      for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
          msg << R(k, m) << ", ";
        }
      }
      for (int k = 0; k < K; k++) {
        for (int l = k; l < K; l++) {
          msg << Q(k, l) << ", ";
        }
      }
      for (int m = 0; m < M; m++) {
        for (int n = m; n < M; n++) {
          msg << T(m, n) << ", ";
        }
      }
      for (int m = 0; m < M; m++) {
        msg << A(m) << ", ";
      }
      for (int k = 0; k < K; k++) {
        msg << v(k) << ", ";
      }
      for (int k = 0; k < K; k++) {
        msg << v_rand(k) << ", ";
      }
    }
    
    std::string msg_str = msg.str();
    msg_str = msg_str.substr(0, msg_str.length() - 2);
    cout << msg_str << endl;
    fprintf(logfile, "%s\n", msg_str.c_str());
    fflush(logfile);
  }
  
  chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  fprintf(logfile, "# Computation took %lld seconds\n",
          chrono::duration_cast<chrono::seconds>(end - begin).count());
  fclose(logfile);

  return 0;
}
