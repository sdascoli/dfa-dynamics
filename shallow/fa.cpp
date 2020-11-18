/*
 * File:    fa.cpp
 *
 * Authors:  Maria Refinetti <mariaref@gmail.com>
 *           Sebastian Goldt <goldt.sebastian@gmail.com>
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

// #define ARMA_NO_DEBUG
#include <armadillo>
#include <chrono>
using namespace arma;

#include "libscmpp.h"

const char * usage = R"USAGE(

This is fa++, a tool to analyse training by (direct) feedback alignment in
two-layer neural networks in the teacher-student setup.

usage: scmpp.exe [-h] [--g G] [-N N] [-M M] [-K K] [--lr LR]
                     [--init INIT] [--steps STEPS] [--uniform A]
                     [-s SEED] [--quiet] [--fa]

optional arguments:

-s, --sigma SIGMA     std. dev. of teacher's output noise. Default=0.
                        For classification, the probability that a label is
                        drawn at random.
  -h, -?                show this help message and exit
  --g G                 activation function for the teacher and student;
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
  -i, --init INIT       weight initialisation:
                          1,2: i.i.d. Gaussians with sd 1 or 1/sqrt(N), resp.
                            3: informed initialisation; only for K \ge M.
                            4: denoising initialisation
                            5: 'mixed': i.i.d. Gaussian with 1/sqrt(N), 1/sqrt(K)
                            11: zero initial weights
  --b_init INIT         initialisation of feedback vector:
                          0: matched to initial second layer weight of the student
                          1: random
                          2: orthogonal to the teacher's second layer weights
                          3: with only positive components
  --stop                generalisation error at which to stop the simulation.
  --store               store initial overlap and final weight matrices.
  -r SEED, --seed SEED  random number generator seed. Default=0
  --dummy               dummy command that doesn't do anything but helps with parallel.
  --quiet               be quiet and don't print order parameters to cout.
)USAGE";

const int NUM_TEST_SAMPLES = 100000;
// initialisation options for the feedback vecto
const int INIT_ZEROS = 11;
const int MATCHED=0;
const int RAND=1;
const int ORTH=2;
const int POS=3;

int main(int argc, char* argv[]) {
  // flags; false=0 and true=1
  int both  = 1;      // train both layers
  int fa    = 0;      // train using feedback alignment
  int b_init    = 1; // random vector initialisation
  int normalise = 0;  // second-layer 1/(node number)
  int meanfield = 0;  // mean-field second layer 1 / sqrt(node number)
  double sigma     = 0;  // std.dev. of the teacher's additive output noise
  int mix = 0;        // alternate the sign of the teacher second layer
  int quiet = 0;      // don't print the order parameters to cout
  int store = 0;      // store initial weights etc.
  int dummy = 0;      // dummy parameter
  // other parameters
  int    g         = ERF;  // teacher activation function
  int    N         = 500;  // number of inputs
  int    M         = 4;  // num of teacher's hidden units
  int    K         = 4;  // num of student's hidden units
  double lr        = 0.2;  // learning rate
  int    init      = INIT_LARGE;  // initial weights
  double stop      = 1e-6;  // value of eg at which to stop the simulation
  double max_steps = 1000;  // max number of gradient updates / N
  int    seed      = 0;  // random number generator seed
  double uniform   = 0; // value of all weights in the teacher's second layer

  // parse command line options using getopt
  int c;

  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"fa",         no_argument, &fa,             1},
    {"normalise",  no_argument, &normalise,      1},
    {"meanfield",  no_argument, &meanfield,      1},
    {"mix",        no_argument, &mix,            1},
    {"store",      no_argument, &store,          1},
    {"dummy",      no_argument, &dummy,          1},
    {"quiet",      no_argument, &quiet,          1},
    {"g",      required_argument, 0, 'g'},
    {"b_init", required_argument, 0, 'b'},
    {"N",       required_argument, 0, 'N'},
    {"M",       required_argument, 0, 'M'},
    {"K",       required_argument, 0, 'K'},
    {"lr",      required_argument, 0, 'l'},
    {"sigma",   required_argument, 0, 's'},
    {"init",    required_argument, 0, 'i'},
    {"uniform", required_argument, 0, 'u'},
    {"stop",    required_argument, 0, 'j'},
    {"steps",   required_argument, 0, 'a'},
    {"seed",    required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };

  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "g:h:N:M:K:l:i:u:j:a:r:b:",
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
      case 'N':
        N = atoi(optarg);
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
      case 'u':  // value of the second layer weights of the teacher
        uniform = atof(optarg);
        break;
      case 'i':  // initialisation of the weights
        init = atoi(optarg);
        break;
      case 'j':  // value of the second layer weights of the teacher
        stop = atof(optarg);
        break;
      case 'a':  // number of steps in multiples of N
        max_steps = atof(optarg);
        break;
      case 'r':  // random number generator seed
        seed = atoi(optarg);
        break;
      case 'b':  // initialisation random vector
        b_init = atoi(optarg);
        break;
      case 's':
        sigma = atof(optarg);
        break;
      case '?':
        cout << usage << endl;
        return 0;
      default:
        abort ();
    }
    
  }
  if(uniform==1 and g==2 ){b_init = POS;}
  
  if (meanfield and normalise) {
    cerr << "Cannot have both meanfield and normalised networks. Will exit now !" << endl;
    return 1;
  }
  
  // set the seed
  arma_rng::set_seed(seed);
  // Draw the weights of the network and their activation functions
  mat B = mat();  // teacher input-to-hidden weights
  vec A = vec();  // teacher hidden-to-output weights
  bool success = init_teacher_randomly(B, A, N, M, uniform, both, normalise,
                                       meanfield, mix);
  arma_rng::set_seed(seed);//Set the seed after generationg the teacher so multiple run with the same teacher :)
  if (!success) {
    // some error happened during teacher init
    cerr << "Could not initialise teacher; will exit now!" << endl;
    return 1;
  }

  mat w = mat(K, N);   // student weights
  vec v = vec(K);
  vec v_rand=vec(K);
  switch (init) {
    case INIT_LARGE:
    case INIT_SMALL:
    case INIT_MIXED:
    case INIT_MIXED_NORMALISE: // intentional fall-through
      init_student_randomly(w, v, N, K, init, uniform, both, normalise, meanfield, false);
      break;
    case INIT_ZEROS: // intentional fall-through
      w = zeros<mat>(K,N);
      v= zeros<vec>(K);
    break;
    case INIT_NATI: {
      w = 1e-3 * randn<mat>(size(w));
      v = 1. / K * ones<vec>(K);
      break;
    }
    case INIT_NATI_MF: {
      w = 1e-3 * randn<mat>(size(w));
      v = 1. / sqrt(K) * ones<vec>(K);
      break;
    }
    default:
      cerr << "Init must be within 1-2, 5-8. Will exit now." << endl;
      return 1;
  }
  cout<<"b init is "<<b_init<<endl;
  int counter;int iter_counter;
  switch (b_init) {
    case MATCHED:
      v_rand = v;
      break;
    case RAND:
      //in order to have teacher andf random vector selected in the same way
      arma_rng::set_seed(0);
      v_rand = randn<vec>(K);
      arma_rng::set_seed(seed);
      break;
    case POS:
      iter_counter=0;
      do {
        v_rand = randu<vec>(K);
//        v_rand.print("feedback vector= ");
//        cout<<"min M K "<<std::min(K,M)<<endl;
        counter=0;
        for(int k;k<K;++k){
          if(v_rand(k)>0){
            counter+=1;}
        }
//        cout<<" counter is "<<counter<<endl;
        iter_counter+=1;
        if(iter_counter>10000000){break;}
      }
      while (counter<std::min(K,M));
//      cout<<"ALL OKAY"<<endl;
      v_rand.print("feedback vector= ");
    break;
    case ORTH:
      for(int k=0;k<K/2;++k){
        v_rand(k)=A(k);
        v_rand(K/2+k)=-A(k);
      }
      break;
  }
  mat (*g1_fun)(mat&);
  mat (*g2_fun)(mat&);
  mat (*dgdx_fun)(mat&);
  switch (g) {  // find the teacher's activation function
    case LINEAR:
      g1_fun = g_lin;
      g2_fun = g_lin;
      dgdx_fun = dgdx_lin;
      break;
    case ERF:
      g1_fun = g_erf;
      g2_fun = g_erf;
      dgdx_fun = dgdx_erf;
      break;
    case RELU:
      g1_fun = g_relu;
      g2_fun = g_relu;
      dgdx_fun = dgdx_relu;
      break;
    case QUAD:
      g1_fun = g_quad;
      g2_fun = g_quad;
      dgdx_fun = dgdx_quad;
      break;
    default:
      cerr << "g has to be linear (g=" << LINEAR << "), erg1 (g=" << ERF <<
          "), ReLU (g=" << RELU << ") or sign (g1=" << SIGN << ") or quad (g="
           << QUAD << "). " << endl;
      cerr << "Will exit now!" << endl;
      return 1;
  }
  
  const char* g1_name = activation_name(g);
  const char* g2_name = activation_name(g);

  std::ostringstream welcome;
  welcome << "# This is fa++" << endl
          << "# g1=" << g1_name << ", g2=" << g2_name
          << ", N=" << N << ", M=" << M << ", K=" << K
          << ", steps/N=" << max_steps << ", seed=" << seed << endl
          << "# lr=" << lr << ", sigma=" << sigma<< endl;
  welcome << "# training both layers";
  if (uniform > 0)
    welcome << " (teacher's second layer has uniform weights=" << uniform << ")";
  welcome << endl;
  
  // find printing times
  vec steps = logspace<vec>(-1, log10(max_steps), 200);

  // generate a finite test set
  mat test_xs = randn<mat>(NUM_TEST_SAMPLES, N);
  
  // we are comparing to the noiseless teacher output, so no noise is addded!
  mat test_ys = phi(B, A, test_xs, g1_fun);
  welcome << "# Generated test set with " << NUM_TEST_SAMPLES << " samples" << endl;

  switch (init) {
    case INIT_SMALL:
      welcome << "# initial weights have small std dev" << endl;
      break;
    case INIT_MIXED:
      welcome << "# initial weights have mixed std dev 1/sqrt(N), 1/sqrt(K)" << endl;
      break;
    case INIT_ZEROS:
      welcome << "# initial weights all set to 0" << endl;
    break;
    case INIT_LARGE:
      welcome << "# initial weights have std dev 1" << endl;
      break;      
    case INIT_INFORMED:
      welcome << "# informed initialisation" << endl;
      break;
  }
  switch (b_init) {
    case RAND:
      welcome << "# random v_rand" << endl;
      break;
    case MATCHED:
      welcome << "# v_rand matched to initial student weight" << endl;
      break;
    case ORTH:
      welcome << "# v_rand orthogonal" << endl;
    break;
  }

  welcome << "# 0 steps / N, 1 eg, 2 et, 3 diff, ";
  int col_idx = 4;
  if (!quiet) {
    for (int k = 0; k < K; k++) {
      for (int m = 0; m < M; m++) {
        welcome << col_idx << " R(" << k << ", " << m << "), ";
        col_idx++;
      }
    }
    for (int k = 0; k < K; k++) {
      for (int l = k; l < K; l++) {
        welcome << col_idx << " W(" << k << ", " << l << "), ";
        col_idx++;
      }
    }
    for (int m = 0; m < M; m++) {
      for (int n = m; n < M; n++) {
        welcome << col_idx << " T(" << m << ", " << n << "), ";
        col_idx++;
      }
    }
    for (int m = 0; m < M; m++) {
      welcome << col_idx << " A(" << m << "), ";
      col_idx++;
    }
    for (int k = 0; k < K; k++) {
      welcome << col_idx << " v(" << k << "), ";
      col_idx++;
    }
    for (int k = 0; k < K; k++) {
      welcome << col_idx << " v_rand(" << k << "), ";
      col_idx++;
    }
  }
  
  std::string welcome_str = welcome.str();
  welcome_str = welcome_str.substr(0, welcome_str.length() - 2);
  cout << welcome_str << endl;

  char* uniform_desc;
  asprintf(&uniform_desc, "u%g_", uniform);
  char* log_fname;
  asprintf(&log_fname,
           "fa%d_%s_%s_both_binit%d_%s%s%s%sN%d_M%d_K%d_lr%g_sigma%g_i%d_steps%g_s%d.dat",
           fa, g1_name, g2_name,b_init,
           (uniform > 0 ? uniform_desc : ""), (mix > 0 ? "mix_" : ""),
           (normalise ? "norm_" : ""), (meanfield ? "mf_" : ""),
           N, M, K, lr,sigma, init, max_steps, seed);
  FILE* logfile = fopen(log_fname, "w");
  fprintf(logfile, "%s\n", welcome_str.c_str());

  // save initial conditions
  if (store) {
    mat Q0 = w * w.t() / N;
    mat R0 = w * B.t() / N;
    mat T0 = B * B.t() / N;

    std::string fname = std::string(log_fname);
    fname.replace(fname.end()-4, fname.end(), "_Q0.dat");
    Q0.save(fname, csv_ascii);
    fname.replace(fname.end()-7, fname.end(), "_R0.dat");
    R0.save(fname, csv_ascii);
    fname.replace(fname.end()-7, fname.end(), "_T0.dat");
    T0.save(fname, csv_ascii);
    fname.replace(fname.end()-7, fname.end(), "_A0.dat");
    A.save(fname, csv_ascii);
    fname.replace(fname.end()-7, fname.end(), "_v0.dat");
    v.save(fname, csv_ascii);
    fname.replace(fname.end()-7, fname.end(), "_vrand.dat");
    v_rand.save(fname, csv_ascii);
  }

  std::clock_t c_start = std::clock();
  auto t_start = std::chrono::high_resolution_clock::now();

  mat gradw = mat(size(w));
  vec gradv = vec(size(v));
  mat dw = zeros<mat>(size(w));
  vec dv = zeros<vec>(size(v));

  double dstep = 1.0 / N;
  uword step_print_next_idx = 0;
  bool done = false;

  // inputs and labels used in an actual sgd step
  int bs = 1;
  mat xs = mat(bs, N);
  mat ys = mat(bs, 1);
  double t = 0;
  while(!done) {
    if (t > steps(step_print_next_idx) || t == 0 ) {
      std::ostringstream msg;

      mat W = w * w.t() / N;
      mat R = w * B.t() / N;
      mat T = B * B.t() / N;

      // compute the TEST error
      
      mat act = test_xs * w.t() / sqrt(w.n_cols);  // (num_test, K)
      mat hidden = (*g2_fun)(act);  // (num_test, K)
      mat derivs = (*dgdx_fun)(act);  // (num_test, K)
      mat y_preds =  hidden * v;  // (num_test, 1)
      mat errors = test_ys - y_preds; // (num_test, 1)
      double eg = as_scalar(mean(pow(errors, 2)));

      if (eg < stop && t > 1000) {
        done = true;
      }
      double diff = datum::nan;

      msg << t << ", " << eg << ", " << datum::nan << ", " << diff;

      if (!quiet) {
        msg<<",";
        for (int k = 0; k < K; k++) {
          for (int m = 0; m < M; m++) {
            msg << R(k, m) << ", ";
          }
        }
        for (int k = 0; k < K; k++) {
          for (int l = k; l < K; l++) {
            msg << W(k, l) << ", ";
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

      while (!done && t > steps(step_print_next_idx)) {
        step_print_next_idx++;
        if (step_print_next_idx == steps.n_elem) {
          done = true;
        }
      }
    }

    xs = randn<mat>(bs, N);  // random input
    ys = phi(B, A, xs, g1_fun);  // teacher label
    //adds output noise to the teacher
    if (sigma>0){
      ys += sigma * randn<mat>(size(ys));}
    
    // forward pass
    mat act = w * xs.t() / sqrt(w.n_cols);  // (K, bs) activation of the hidden units
    mat hidden = (*g2_fun)(act);  // (K, bs) apply the non-linearity point-wise
    mat ys_pred = v.t() * hidden;  // (1, bs) and sum up!

    // backward pass
    vec error = ys - ys_pred.t();  // (bs, 1)
    mat deriv = dgdx_fun(act);  // (K, bs)
    deriv.each_col() %= (fa ? v_rand : v); //    if fa is true use the random weights else use the true second layer weights of the student

    gradv = 1. / bs * g2_fun(act) * error;
    gradw = 1. / bs * deriv * diagmat(error) * xs;    

    v += lr / N  * gradv;
    w += lr / sqrt(N) * gradw;
    t += dstep;
  }

  std::clock_t c_end = std::clock();
  auto t_end = std::chrono::high_resolution_clock::now();
  std::ostringstream time_stream;
  time_stream << "# CPU time used: "
             << (c_end-c_start) / CLOCKS_PER_SEC << " s\n"
              << "# Wall clock time passed: "
             << std::chrono::duration_cast<std::chrono::seconds>(t_end-t_start).count()
              << " s\n";
  std::string time_string = time_stream.str();
  cout << time_string;
  fprintf(logfile, "%s", time_string.c_str());
  fclose(logfile);

  if (store) {    // store the final teacher/student weights
    std::string fname = std::string(log_fname);
    fname.replace(fname.end()-4, fname.end(), "_w.dat");
    w.save(fname.c_str(), csv_ascii);
    fname.replace(fname.end()-6, fname.end(), "_v.dat");
    v.save(fname.c_str(), csv_ascii);
  }
  return 0;
}
