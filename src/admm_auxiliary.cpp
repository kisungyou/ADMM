#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


/*
 * Auxiliary 1. multiple inveresion
 */
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
arma::cube multipleinversion(arma::mat A, double rho, arma::mat L, arma::mat R, arma::vec lambda2){
  // 1. get size information
  const int n = A.n_rows;
  const int p = A.n_cols;
  const int d1 = R.n_rows;
  const int d2 = L.n_rows;
  const int nlambda = lambda2.n_elem;

  // 2. initialize and preliminary computations
  arma::cube output(p,p,nlambda,fill::zeros);

  arma::mat AA = A.t()*A;
  arma::mat LL = L.t()*L;
  arma::mat RR = R.t()*R; // equivalent to M = R^T R;

  // 3. computation : the very first one
  output.slice(0) = arma::pinv(AA+(rho*LL)+(2*lambda2(0)*RR));

  // 4. computation : iterative update
  double dlbd = 0.0;                     // difference in lambda2 values
  arma::mat Ainv(p,p,fill::zeros);       // inverse of A for previous step
  arma::mat solveLHS(d2,d2,fill::zeros); // iterative comp::inverse term
  arma::mat solveTMP(d2,p,fill::zeros);  // iterative comp::solveLHS*R
  arma::mat eye_d2(d2,d2,fill::eye);     // diagonal matrix of size (d2-by-d2)
  arma::mat eye_p(p,p,fill::eye);        // diagnoal matrix of size (p -by- p)

  arma::mat invold = output.slice(0);
  arma::mat invnew(p,p,fill::zeros);

  for (int i=1;i<nlambda;i++){
    // 4-1. update setup
    dlbd = lambda2(i)-lambda2(i-1);

    // 4-2. preliminary computation
    solveLHS = (eye_d2 + (2*dlbd*(R*invold*R.t())));
    solveTMP = arma::solve(solveLHS, R);

    // 4-3. compute and record
    invnew = (eye_p - 2*dlbd*(invold*R.t()*solveTMP))*invold;
    output.slice(i) = invnew;

    // 4-4. update
    invold = invnew;
  }

  // 5. return output
  return(output);
}
