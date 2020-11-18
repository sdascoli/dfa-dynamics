/*
 * File:    bp_ode.cpp
 *
 * Author: Code taken from https://arxiv.org/pdf/1906.08632.pdf
 *         Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Version: 0.2
 *
 * Date:    December 2018
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

const int NUM_DATAPOINTS = 300;

const char * usage = R"USAGE(
This tool integrates the equations of motion that describe the generalisation
dynamics of two-layer neural networks with sigmoidal activation funtion.
usage: scmpp_ode.exe [-h] [-M M] [-K K] [--lr LR] [--lr2 LR2] [--sigma SIGMA]
                        [--wd WD] [--overlap OVERLAP] [--dt DT] [--steps STEPS]
                        [--quiet] [--both] [--uniform A]
optional arguments:
  -h, -?                show this help message and exit
  --g G                 activation function for teacher and student;
                           0-> linear, 1->erf.
  -M, --M M             number of hidden units in the teacher network
  -K, --K K             number of hidden units in the student network
  -l, --lr LR           learning rate
  --lr2 LR2             learning rate for the second layer only. If not
                          specified, we will use the same learning rate for
                          both layers.
  -s, --sigma SIGMA     std. dev. of teacher's output noise. Default=0.
                          For classification, the probability that a label is
                          drawn at random.
  -w, --wd WD           weight decay constant. Default=0.
  -a, --steps STEPS     max. weight update steps in multiples of N
  --init INIT           weight initialisation:
                           1: large initial weights, with initial overlaps from --overlaps
                           2: small initial weights
                           3: informed initialisation; only for K \ge M.
                           4: denoising
  --prefix              file prefix to load initial conditions from
  --both                train both layers.
  --uniform A           make all of the teacher's second layer weights equal to
                          this value.
  --overlap OVERLAP     initial overlap between teacher, student vectors
  --dt DT               integration time-step
  -r SEED, --seed SEED  random number generator seed. Default=0
  -q --quiet            be quiet and don't print order parameters to cout.
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
               mat& Q, mat& R, mat& T, vec& A, vec& v, double(*J2)(mat&),
               double(*I2)(mat&), double(*I3)(mat&), double(*I4)(mat&),
               double lr, double lr2, double wd, double sigma, bool both) {
  int K = R.n_rows;
  int M = R.n_cols;

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
        R(i, n) -= dt * wd * R(i, n);

        for (int m = 0; m < M; m++) { // teacher
          update_C3(C3, C, i, K+n, K+m);
          R(i, n) += dt * lr * v(i) * A(m) * I3(C3);
        }
        for (int j  = 0; j < K; j++) {  // student
          update_C3(C3, C, i, K+n, j);
          R(i, n) -= dt * lr * v(i) * v(j) * I3(C3);
        }
      }
    }

    // integrate Q
    for (int i = 0; i < K; i++) {  // student
      for (int k = i; k < K; k++) {  // student
        // weight decay
        Q(i, k) -= dt * 2 * wd * Q(i, k);

        // terms proportional to the learning rate
        for (int m = 0; m < M; m++){ // teacher
          update_C3(C3, C, i, k, K + m);
          Q(i, k) += dt * lr * v(i) * A(m) * I3(C3);
          update_C3(C3, C, k, i, K + m);
          Q(i, k) += dt * lr * v(k) * A(m) * I3(C3);
        }
        for (int j = 0; j < K; j++) { // student
          update_C3(C3, C, i, k, j);
          Q(i, k) -= dt * lr * v(i) * v(j) * I3(C3);
          update_C3(C3, C, k, i, j);
          Q(i, k) -= dt * lr * v(k) * v(j) * I3(C3);
        }

        // noise term
        if (sigma > 0) {
          //Q(i, k) += dt * v(i) * v(k) * (pow(lr, 2) * pow(sigma, 2) * 2 / datum::pi /
          //                 sqrt(1+Q(i, i)+Q(k, k)-pow(Q(i, k), 2)+Q(i, i)*Q(k, k)));
          // Q(i, k) += dt * v(i) * v(k) * (pow(lr, 2) * pow(sigma, 2));
          update_C2(C2, C, i, k);
          Q(i, k) += dt * v(i) * v(k) * (pow(lr, 2) * pow(sigma, 2)) * J2(C2);
        }
          
        // SGD terms quadratic to the learning rate squared
        for (int n = 0; n < M; n++) {  // teacher
          for (int m = 0; m < M; m++) {  // teacher
            update_C4(C4, C, i, k, K + n, K + m);
            Q(i, k) += dt * pow(lr, 2) * v(i) * v(k) * A(n) * A(m) * I4(C4);
          }
        }

        for (int j = 0; j < K; j++) {  // student
          for (int n = 0; n < M; n++) {  // teacher
            update_C4(C4, C, i, k, j, K + n);
            Q(i, k) -= dt * pow(lr, 2) * v(i) * v(k) * v(j) * A(n) * 2 * I4(C4);
          }
        }

        for (int j = 0; j < K; j++) {  // student
          for (int l = 0; l < K; l++) {  // student
            update_C4(C4, C, i, k, j, l);
            Q(i, k) += dt * pow(lr, 2) * v(i) * v(k) * v(j) * v(l) * I4(C4);
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
        v_new(i) -= dt * wd * v(i);

        for (int k = 0; k < K; k++) {  // student
          update_C2(C2, C, i, k);
          v_new(i) -= dt * lr2 * v(k) * I2(C2);
        }

        for (int n = 0; n < M; n++) { // teacher
          update_C2(C2, C, i, K + n);
          v_new(i) += dt * lr2 * A(n) * I2(C2);
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
  int quiet = 0;  // don't print the order parameters to cout
  int both = 0;
  // other parameters
  int    g         = ERF;  // teacher and student activation function
  int    M         = 4;  // num of teacher's hidden units
  int    K         = 4;  // num of student's hidden units
  double lr        = 0.5;  // learning rate
  double lr2       = -1;  // learning rate for the second layer.
  double wd        = 0;  // weigtht decay constant
  double sigma     = 0;  // std.dev. of the teacher's additive output noise
  double dt        = 0.01;
  int    init      = INIT_LARGE; // initialisation
  double uniform   = 0; // value of all weights in the teacher's second layer
  double initial_overlap = 1e-9;  // initial weights
  string prefix;  // file name prefix to preload the weights
  double max_steps = 1000;  // max number of gradient updates / N
  int    seed      = 0;  // random number generator seed

  // parse command line options using getopt
  int c;

  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"quiet",      no_argument, &quiet,          1},
    {"both",       no_argument, &both,           1},
    {"g",       required_argument, 0, 'g'},
    {"M",       required_argument, 0, 'M'},
    {"K",       required_argument, 0, 'K'},
    {"lr",      required_argument, 0, 'l'},
    {"lr2",     required_argument, 0, 'm'},
    {"sigma",   required_argument, 0, 's'},
    {"wd",      required_argument, 0, 'w'},
    {"dt",      required_argument, 0, 'd'},
    {"init",    required_argument, 0, 'i'},
    {"prefix",  required_argument, 0, 'f'},
    {"uniform", required_argument, 0, 'u'},
    {"overlap", required_argument, 0, 'o'},
    {"steps",   required_argument, 0, 'a'},
    {"seed",    required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };

  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "g:M:K:l:s:w:c:a:r:o:u:",
                    long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) {
      break;
    }

    switch (c) {
      case 0:
        break;
      case 'g':
        g = atoi(optarg);
        break;
      case 'M':
        M = atoi(optarg);
        break;
      case 'K':
        K = atoi(optarg);
        break;
      case 'l':
        lr = atof(optarg);
        break;
      case 'm':
        lr2 = atof(optarg);
        break;
      case 's':
        sigma = atof(optarg);
        break;
      case 'w':
        wd = atof(optarg);
        break;
      case 'i':  // initialisation of the weights
        init = atoi(optarg);
        break;
      case 'f':  // pre-load initial conditions from file with this prefix
        prefix = string(optarg);
        break;
      case 'o':  // initial overlap
        initial_overlap = atof(optarg);
        break;
      case 'u':  // value of the second layer weights of the teacher
        uniform = atof(optarg);
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
      case 'h':  // intentional fall-through
      case '?':
        cout << usage << endl;
        return 0;
      default:
        abort ();
    }
  }

  // if not explicitly given, use the same learning rate in both layers
  if (lr2 < 0) {
    lr2 = lr;
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
  
  FILE* logfile;
  const char* g_name = activation_name(g);
  if (prefix.empty()) {
    char* uniform_desc;
    asprintf(&uniform_desc, "u%g_", uniform);
    char* io_desc;
    asprintf(&io_desc, "io%g_", initial_overlap);
    char* lr2_desc;
    asprintf(&lr2_desc, "2lr%g_", lr2);
    char* log_fname;
    asprintf(&log_fname,
             "scmpp_ode_%s_%s_%s%sM%d_K%d_lr%g_%swd%g_sigma%g_i%d_%ssteps%g_dt%g_s%d.dat",
             g_name, g_name, (both ? "both_" : ""), (abs(uniform) > 0 ? uniform_desc : ""),
             M, K, lr, (lr2 != lr ? lr2_desc : ""), wd, sigma, init,
             (prefix.empty() ? io_desc : ""), max_steps, dt, seed);
    logfile = fopen(log_fname, "w");
  } else {
    string log_fname = prefix;
    log_fname.append("_ode.dat");
    logfile = fopen(log_fname.c_str(), "w");
  }

  ostringstream welcome;
  welcome << "# This is scm++ ODE integrator for two-layer NN" << endl
          << "# g1=g2=" << g_name << ", M=" << M << ", K=" << K
          << ", steps/N=" << max_steps << ", seed=" << seed << endl
          << "# lr=" << lr << ", lr2=" << lr2 << ", sigma=" << sigma << ", wd=" << wd
          << ", dt " << dt << ", initial overlap=" << initial_overlap << endl;
  if (both) {
    welcome << "# training both layers";
    if (uniform)
      welcome << " (second layer has uniform weights)";
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
  if (abs(uniform) > 0) {
    A *= uniform;
  } else if (both) {
    A = vec(M, fill::randn);
  }

  if (!prefix.empty()) {
    prefix.append("_Q0.dat");
    bool ok = Q.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_R0.dat");
    ok = ok && R.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_T0.dat");
    ok = ok && T.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_A0.dat");
    ok = ok && A.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_v0.dat");
    ok = ok && v.load(prefix);
    if (!ok) {
      cerr << "Error loading initial conditions from files, will exit !" << endl;
      return 1;
    }
  } else if (init == INIT_LARGE) {
    Q = eye<mat>(K, K) + initial_overlap * randn<mat>(K, K);
    // make sure Q is symmetric
    Q = symmatu(Q);
    // overlap between the kth student and the mth teacher
    R = initial_overlap * randn<mat>(K, M);
    v = ones<vec>(K) + initial_overlap * randn<vec>(K);
    if (abs(uniform) > 0 and !both) {
      v.fill(uniform);
    }
  } else if (init == INIT_SMALL) {
    Q = 1. / 2000 * eye<mat>(K, K) + initial_overlap * randn<mat>(K, K);
    // make sure Q is symmetric
    Q = symmatu(Q);
    // overlap between the kth student and the mth teacher
    R = initial_overlap * randn<mat>(K, M);
    v = 1. / 2000 * ones<vec>(K) + initial_overlap * randn<vec>(K);
    if (abs(uniform) > 0 and !both) {
      v.fill(uniform);
    }
  } else if (init == INIT_INFORMED) {
    if (K < M) {
      cerr << "Cannot do an informed initialisation for K < M " << endl
           << "Will exit now!" << endl;
      return 1;
    }
    Q = initial_overlap * randn<mat>(K, K);
    Q.submat(0, 0, M-1, M-1) += eye<mat>(M, M);
    Q = symmatu(Q);
    R = initial_overlap * randn<mat>(K, M);
    R.submat(0, 0, M-1, M-1) += eye<mat>(M, M);
    v = initial_overlap * randn<vec>(K);
    v.subvec(0, M-1) += A;
  } else if (init == INIT_DENOISE) {
      if (K < M) {
        cerr << "Cannot do a denoiser initialisation with K<M." << endl
             << "Will exit now !" << endl;
        return 1;
      }
      if (!both) {
        cerr << "Need to be able to change the second-layer weights to do a "
             << "denoiser initialisation. Will exit now !" << endl;
        return 1;
      }
    // identity + background noise
    Q = eye<mat>(K, K) + initial_overlap * randn<mat>(K, K);
    for (int i = 0; i < K; i++) {
      for (int k = i + 1; k < K; k++) {
        if ((i % M) == (k % M)) {
          Q(i, k) += 1;
        }
      }
    }
    Q = symmatu(Q);
    R = initial_overlap * randn<mat>(K, M);
    for (int i = 0; i < K; i++) {
      R(i, i % M) += 1;
    }
    for (int k = 0; k < K; k++) {
      v(k) = A(k % M);
      // now do the proper rescaling to achieve averaging:
      v(k) = (k % M) <= (K % M - 1) ? v(k)/(floor(K/M) + 1) : v(k)/floor(K/M);
    }
  } else {
    cerr << "--init must be 1 (random init) or 2 (file) or 3 (informed init) "
         << " or 4 (denoising). " << endl << "Will exit now !" << endl;
    return 1;
  }

  // find printing times
  vec print_times = logspace<vec>(-1, log10(max_steps), NUM_DATAPOINTS);
  vec durations = diff(print_times);

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  double t = 0;
  bool converged = false;
  for (double& duration : durations) {
    double eg = eg_analytical(Q, R, T, A, v, g_fun, g_fun);
    string msg = status(t, eg, datum::nan, datum::nan, Q, R, T, A, v);
    cout << msg << endl;
    fprintf(logfile, "%s\n", msg.c_str());
    fflush(logfile);

    if (eg < 1e-14 && t > 100) {
      converged = true;
      break;
    } else {
      propagate(duration, dt, t, Q, R, T, A, v,
                J2_fun, I2_fun, I3_fun, I4_fun,
                lr, lr2, wd, sigma, both);
    }
  }
  if (!converged) {
    double eg = eg_analytical(Q, R, T, A, v, g_fun, g_fun);
    string msg = status(t, eg, datum::nan, datum::nan, Q, R, T, A, v);
    cout << msg << endl;
    fprintf(logfile, "%s\n", msg.c_str());
    fflush(logfile);
  }
  
  chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  fprintf(logfile, "# Computation took %lld seconds\n",
          chrono::duration_cast<chrono::seconds>(end - begin).count());
  fclose(logfile);

  return 0;
}
