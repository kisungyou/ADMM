#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

arma::vec spca_gamma(arma::vec sigma, double r){
  const int p = sigma.n_elem;
  int indj = 0;
  double term1 = 0.0;
  double term2 = 0.0;
  for (int j=0;j<p;j++){
    term1 = sigma(j);
    for (int k=j;k<p;k++){
      term2 += sigma(k);
    }
    term2 = (term2-r)/(p-j);
    if (term1 > term2){
      indj = j;
      break;
    }
  }
  double theta = 0.0;
  for (int j=indj;j<p;j++){
    theta += sigma(j);
  }
  theta = (theta-r)/(p-indj);

  arma::vec output(p,fill::zeros);
  for (int i=0;i<p;i++){
    term1 = sigma(i)-theta;
    if (term1>0){
      output(i) = term1;
    }
  }
  return(output);
}

arma::mat spca_shrinkage(arma::mat A, const double tau){
  const int n = A.n_rows;
  arma::mat output(n,n,fill::zeros);
  double zij    = 0.0;
  double abszij = 0.0;
  double signer = 0.0;
  for (int i=0;i<n;i++){
    for (int j=0;j<n;j++){
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


/*
 * sparse pca
 */
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
Rcpp::List admm_spca(const arma::mat& Sigma, const double reltol, const double abstol,
                     const int maxiter, double mu, double rho){
  // 1. get parameters
  int p = Sigma.n_cols;

  // 2. set updating objects
  arma::mat Xold(p,p,fill::zeros);
  arma::mat Xnew(p,p,fill::zeros);
  arma::mat Yold(p,p,fill::zeros);
  arma::mat Ynew(p,p,fill::zeros);
  arma::mat Lold(p,p,fill::zeros);
  arma::mat Lnew(p,p,fill::zeros);

  arma::mat costX(p,p,fill::zeros);
  arma::mat costY(p,p,fill::zeros);
  arma::vec eigval(p,fill::zeros);   // for EVD of costX
  arma::mat eigvec(p,p,fill::zeros);

  // 3. iteration records
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  // 4. main iteration
  int k=0;
  double ythr = mu*rho;
  double normX = 0.0;
  double normY = 0.0;
  for (k=0;k<maxiter;k++){
    // 4-1. update 'X'
    costX = Yold + mu*Lold + mu*Sigma;
    eig_sym(eigval, eigvec, costX);
    arma::vec gamma = spca_gamma(eigval, 1.0);
    Xnew = eigvec*arma::diagmat(gamma)*eigvec.t();

    // 4-2. update 'Y'
    costY = Xnew-mu*Lold;
    Ynew  = spca_shrinkage(costY, ythr);

    // 4-3. update 'L'
    Lnew  = Lold - (Xnew-Ynew)/mu;

    // 4-4. diagnostics for reporting
    h_r_norm(k) = arma::norm(Xnew-Ynew,"fro");
    h_s_norm(k) = arma::norm(Yold-Ynew,"fro")/mu;

    normX = arma::norm(Xnew,"fro");
    normY = arma::norm(Ynew,"fro");
    if (normX >= normY){
      h_eps_pri(k) = p*abstol + reltol*normX;
    } else {
      h_eps_pri(k) = p*abstol + reltol*normY;
    }
    h_eps_dual(k) = p*abstol + reltol*arma::norm(Lnew,"fro");

    // 4-5. updating and termination
    Xold = Xnew;
    Yold = Ynew;
    Lold = Lnew;

    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))){
      break;
    }
  }

  // 5. report results
  List output;
  output["X"] = Xold;             // coefficient function
  output["k"] = k;             // number of iterations
  output["r_norm"] = h_r_norm;
  output["s_norm"] = h_s_norm;
  output["eps_pri"] = h_eps_pri;
  output["eps_dual"] = h_eps_dual;
  return(output);
}
