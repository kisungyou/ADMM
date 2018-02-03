#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

arma::vec shrink_vec_rpca(arma::vec x, double tau){
  const int n = x.n_elem;
  arma::vec output(n,fill::zeros);
  double xij    = 0.0;
  double absxij = 0.0;
  double signer = 0.0;
  for (int i=0;i<n;i++){
    xij = x(i);
    if (xij >= 0){
      signer = 1.0;
      absxij = xij;
    } else {
      signer = 1.0;
      absxij = -xij;
    }
    if (absxij > tau){
      output(i) = signer*(absxij-tau);
    }
  }
  return(output);
}
arma::mat shrink_mat_rpca(arma::mat A, const double tau){
  const int n = A.n_rows;
  const int p = A.n_cols;
  arma::mat output(n,p,fill::zeros);
  double zij    = 0.0;
  double abszij = 0.0;
  double signer = 0.0;
  for (int i=0;i<n;i++){
    for (int j=0;j<p;j++){
      zij = A(i,j);
      if (zij >= 0){
        signer = 1.0;
        abszij = zij;
      } else {
        signer = -1.0;
        abszij = -zij;
      }

      if (abszij > tau){
        output(i,j) = signer*(abszij-tau);
      }
    }
  }
  return(output);
}
arma::mat rpca_vectorpadding(arma::vec x, const int n, const int p){
  arma::mat output(n,p,fill::zeros);
  if (n<p){
    for (int i=0;i<n;i++){
      output(i,i) = x(i);
    }
  } else {
    for (int j=0;j<p;j++){
      output(j,j) = x(j);
    }
  }
  return(output);
}


/*
 * robust pca
 */
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
Rcpp::List admm_rpca(const arma::mat& M, const double tol, const int maxiter,
                     double mu, double lambda){
  // 1. get parameters
  const int n1 = M.n_rows;
  const int n2 = M.n_cols;

  double invmu = 1/mu;
  double lbdmu = lambda/mu;

  // 2. set updating objects
  arma::mat Lold(n1,n2,fill::zeros);
  arma::mat Lnew(n1,n2,fill::zeros);
  arma::mat Sold(n1,n2,fill::zeros);
  arma::mat Snew(n1,n2,fill::zeros);
  arma::mat Yold(n1,n2,fill::zeros);
  arma::mat Ynew(n1,n2,fill::zeros);

  arma::mat costL(n1,n2,fill::zeros);
  arma::mat costS(n1,n2,fill::zeros);
  arma::mat costY(n1,n2,fill::zeros);
  arma::mat spadding(n1,n2,fill::zeros);

  arma::mat svdU;
  arma::vec svds;
  arma::mat svdV;
  arma::vec vecshrinkage;

  // 3. iteration records
  arma::vec vectolerance(maxiter,fill::zeros);

  // 4. main iteration
  int k=0;
  double norm1 = 0.0;                 // error LHS term
  double norm2 = arma::norm(M,"fro"); // error RHS term
  double normratio = 0.0;
  for (k=0;k<maxiter;k++){
    //  4-1. update L
    costL = (M - Sold + Yold*invmu);                   // compute term to be decomposed
    svd(svdU, svds, svdV, costL);                      // svd decomposition
    vecshrinkage = shrink_vec_rpca(svds, invmu);       // do shrinkage on singular vector
    spadding = rpca_vectorpadding(vecshrinkage,n1,n2); // we need zero padding on diagmat one
    Lnew = svdU*spadding*svdV.t();                     // update L

    // 4-2. update S
    costS = (M-Lnew+Yold*invmu);                // compute term to be shrinked
    Snew  = shrink_mat_rpca(costS, lbdmu);        // update S

    // 4-3. update Y
    Ynew  = Yold + mu*(M-Lnew-Snew);

    // 4-4. compute error
    norm1     = arma::norm(M-Lnew-Snew,"fro");
    normratio = norm1/norm2;
    vectolerance(k) = normratio;

    // 4-5. updating and termination
    Lold = Lnew;
    Sold = Snew;
    Yold = Ynew;

    if (normratio < tol){
      break;
    }
  }

  // 5. report results
  List output;
  output["L"] = Lold;
  output["S"] = Sold;
  output["k"] = k;
  output["errors"] = vectolerance;
  return(output);
}
