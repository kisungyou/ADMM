#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

arma::colvec tv_shrinkage(arma::colvec a, const double kappa){
  const int n = a.n_elem;
  arma::colvec y(n,fill::zeros);
  for (int i=0;i<n;i++){
    // first term : max(0, a-kappa)
    if (a(i)-kappa > 0){
      y(i) = a(i)-kappa;
    }
    // second term : -max(0, -a-kappa)
    if (-a(i)-kappa > 0){
      y(i) = y(i) + a(i) + kappa;
    }
  }
  return(y);
}

double tv_objective(arma::colvec b, const double lambda, arma::mat D,
                    arma::colvec x, arma::colvec z){
  return(pow(norm(x-b),2)/2 + lambda*norm(z,1));
}

/*
* Total Variation Minimization via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
Rcpp::List admm_tv(const arma::colvec& b, arma::colvec& xinit, const double lambda,
                   const double reltol, const double abstol, const int maxiter,
                   const double rho, const double alpha){
  // 1. get parameters and tv difference matrix
  const int n = b.n_elem;
  arma::vec onesN(n,fill::ones);
  arma::mat D = diagmat(onesN);
  for (int i=0;i<(n-1);i++){
    D(i,i+1) = -1;
  }
  arma::mat I = diagmat(onesN);
  arma::mat DtD = D.t()*D;

  // 2. set ready
  arma::colvec x(n,fill::zeros);
  arma::colvec z(n,fill::zeros);
  arma::colvec u(n,fill::zeros);
  arma::colvec zold(n,fill::zeros);
  arma::colvec Ax_hat(n,fill::zeros);

  // 3. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  double sqrtn = sqrt(static_cast<double>(n));
  arma::vec compare2(2,fill::zeros);
  int k;
  for (k=0;k<maxiter;k++){
    // 3-1. update 'x'
    x = solve(I+rho*DtD, b+rho*D.t()*(z-u));

    // 3-2. update 'z' with relaxation
    zold = z;
    Ax_hat = alpha*D*x + (1-alpha)*zold;
    z = tv_shrinkage(Ax_hat+u, lambda/rho);

    // 3-3. update 'u'
    u = u + Ax_hat - z;

    // 3-4.. dianostics, reporting
    h_objval(k) = tv_objective(b,lambda,D,x,z);
    h_r_norm(k) = norm(D*x-z);
    h_s_norm(k) = norm(-rho*D.t()*(z-zold));

    compare2(0) = norm(D*x);
    compare2(1) = norm(-z);

    h_eps_pri(k)  = sqrtn*abstol + reltol*max(compare2);
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*D.t()*u);

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))){
      break;
    }
  }

  // 5. report results
  List output;
  output["x"] = x;             // coefficient function
  output["objval"] = h_objval; // |x|_1
  output["k"] = k;             // number of iterations
  output["r_norm"] = h_r_norm;
  output["s_norm"] = h_s_norm;
  output["eps_pri"] = h_eps_pri;
  output["eps_dual"] = h_eps_dual;
  return(output);
}
