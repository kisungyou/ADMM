#include <cmath>
#include "RcppArmadillo.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

arma::colvec enet_shrinkage(arma::colvec a, const double kappa){
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

double enet_objective(const arma::mat &A, const arma::colvec &b, const double lambda, const double alpha, const arma::colvec &x, const arma::colvec &z){
  //return(pow(norm(A*x-b,2),2)/2 + lambda*alpha*norm(z,1) + 0.5*(1-alpha)*lambda*pow(norm(x,2),2));
  return(norm(A*x-b,2)/2+lambda*alpha*norm(z,1)+0.5*(1-alpha)*lambda*norm(x,2));
}

arma::mat enet_factor(arma::mat A, double rho){
  const int m = A.n_rows;
  const int n = A.n_cols;
  arma::mat U;
  arma::vec onesN(n,fill::ones);
  U = chol(A.t()*A + rho*diagmat(onesN));
  return(U);
}


//' @keywords internal
//' @noRd
// [[Rcpp::export]]
Rcpp::List admm_enet(const arma::mat& A, const arma::colvec& b,  const double lambda, const double alpha, const double reltol, const double abstol, const int maxiter, const double rho){
  // 1. get parameters
  const int m = A.n_rows;
  const int n = A.n_cols;
  double gamma = lambda*(1-alpha)+rho;

  // 2. set ready
  arma::colvec x(n,fill::zeros);
  arma::colvec z(n,fill::zeros);
  arma::colvec u(n,fill::zeros);
  arma::colvec q(n,fill::zeros);
  arma::colvec zold(n,fill::zeros);
  arma::colvec x_hat(n,fill::zeros);

  // 3. precompute static variables for x-update and factorization
  arma::mat Atb = A.t()*b;
  arma::mat U   = enet_factor(A,gamma); // returns upper
  arma::mat L   = U.t();


  // 4. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  // double rho2 = rho*rho;

  double sqrtn = sqrt(static_cast<double>(n));
  int k;
  for (k=0; k<maxiter; k++){
    // 4-1. update 'x'
    q = Atb + rho*(z-u); // temporary value
    x = solve(trimatu(U),solve(trimatl(L),q));

    // 4-2. update 'z'
    zold = z;
    z = enet_shrinkage(x + u, lambda*alpha/rho);

    // 4-3. update 'u'
    u = u + x - z;

    // 4-3. dianostics, reporting
    h_objval(k) = enet_objective(A,b,lambda,alpha,x,z);
    h_r_norm(k) = arma::norm(x-z);
    h_s_norm(k) = arma::norm(-rho*(z-zold));
    if (norm(x)>norm(-z)){
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(x);
    } else {
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(-z);
    }
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*u);
    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))){
      break;
    }
  }
  // 5. report results
  Rcpp::List output;
  output["x"] = x;             // coefficient function
  output["objval"] = h_objval; // |x|_1
  output["k"] = k;             // number of iterations
  output["r_norm"] = h_r_norm;
  output["s_norm"] = h_s_norm;
  output["eps_pri"] = h_eps_pri;
  output["eps_dual"] = h_eps_dual;
  return(output);

}
