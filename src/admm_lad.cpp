#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

arma::colvec lad_shrinkage(arma::colvec a, const double kappa){
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

/*
* LAD via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
Rcpp::List admm_lad(const arma::mat& A, const arma::colvec& b, arma::colvec& xinit,
                    const double reltol, const double abstol, const int maxiter,
                    const double rho, const double alpha){
  // 1. get parameters
  const int m = A.n_rows;
  const int n = A.n_cols;

  // 2. set ready
  arma::colvec x(n,fill::zeros);
  arma::colvec z(m,fill::zeros);
  arma::colvec u(m,fill::zeros);
  arma::colvec zold(m,fill::zeros);
  arma::colvec Ax_hat(m,fill::zeros);

  // 3. precompute static variables for x-update
  arma::mat R = arma::chol(A.t()*A);

  // 4. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  double sqrtn = sqrt(static_cast<double>(n));
  double sqrtm = sqrt(static_cast<double>(m));
  arma::vec compare3(3,fill::zeros);
  int k;
  for (k=0;k<maxiter;k++){
    // 4-1. update 'x'
    x = solve(trimatu(R),solve(trimatl(R.t()),A.t()*(b+z-u)));

    // 4-2. update 'z' with relaxation
    zold = z;
    Ax_hat = alpha*A*x + (1-alpha)*(zold + b);
    z = lad_shrinkage(Ax_hat - b + u, 1/rho);
    u = u + (Ax_hat - z - b);

    // 4-3. dianostics, reporting
    h_objval(k) = norm(x,1);
    h_r_norm(k) = norm(A*x-z-b);
    h_s_norm(k) = norm(-rho*A.t()*(z-zold));

    compare3(0) = norm(A*x);
    compare3(1) = norm(-z);
    compare3(2) = norm(b);

    h_eps_pri(k) = sqrtm*abstol + reltol*max(compare3);
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*A.t()*u);

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


