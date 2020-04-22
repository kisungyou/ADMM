#' Sparse PCA
#'
#' @description Sparse Principal Component Analysis aims at finding a sparse vector by solving
#' \deqn{\textrm{max}_x~x^T\Sigma x \quad \textrm{s.t.} \quad \|x\|_2\le 1,~\|x\|_0\le K}
#' where \eqn{\|x\|_0} is the number of non-zero elements in a vector \eqn{x}. A convex relaxation
#' of this problem was proposed to solve the following problem,
#' \deqn{\textrm{max}_X~<\Sigma,X> ~\textrm{s.t.} \quad Tr(X)=1,~\|X\|_0 \le K^2, ~X\ge 0,~\textrm{rank}(X)=1}
#' where \eqn{X=xx^T} is a \eqn{(p\times p)} matrix that is outer product of a vector \eqn{x} by itself,
#' and \eqn{X\ge 0} means the matrix \eqn{X} is positive semidefinite.
#' With the rank condition dropped, it can be restated as
#' \deqn{\textrm{max}_X~ <\Sigma,X>-\rho\|X\|_1 \quad \textrm{s.t.}\quad Tr(X)=1,X\ge 0.}
#' After acquiring each principal component vector, an iterative step based on Schur complement deflation method
#' is applied to regress out the impact of previously-computed projection vectors. It should be noted that
#' those sparse basis may \emph{not be orthonormal}.
#'
#' @param Sigma a \eqn{(p\times p)} (sample) covariance matrix.
#' @param numpc number of principal components to be extracted.
#' @param mu an augmented Lagrangian parameter.
#' @param rho a regularization parameter for sparsity.
#' @param abstol absolute tolerance stopping criterion.
#' @param reltol relative tolerance stopping criterion.
#' @param maxiter maximum number of iterations.
#'
#' @return a named list containing \describe{
#' \item{basis}{a \eqn{(p\times numpc)} matrix whose columns are sparse principal components.}
#' \item{history}{a length-\code{numpc} list of dataframes recording iteration numerics. See the section for more details.}
#' }
#'
#' @section Iteration History:
#' For SPCA implementation, main computation is sequentially performed for each projection vector. The \code{history}
#' field is a list of length \code{numpc}, where each element is a data frame containing iteration history recording
#' following fields over iterates,
#' \describe{
#' \item{r_norm}{norm of primal residual}
#' \item{s_norm}{norm of dual residual}
#' \item{eps_pri}{feasibility tolerance for primal feasibility condition}
#' \item{eps_dual}{feasibility tolerance for dual feasibility condition}
#' }
#' In accordance with the paper, iteration stops when both \code{r_norm} and \code{s_norm} values
#' become smaller than \code{eps_pri} and \code{eps_dual}, respectively.
#'
#' @examples
#' ## generate a random matrix and compute its sample covariance
#' X    = matrix(rnorm(1000*5),nrow=1000)
#' covX = stats::cov(X)
#'
#' ## compute 3 sparse basis
#' output = admm.spca(covX, 3)
#'
#' @references
#' \insertRef{ma_alternating_2013a}{ADMM}
#'
#' @export
admm.spca <- function(Sigma, numpc, mu=1.0, rho=1.0, abstol=1e-4, reltol=1e-2, maxiter=1000){
  # -----------------------------------------------------------------
  ## PREPROCESSING
  #   1. data
  if ((!check_data_matrix(Sigma))||(!isSymmetric(Sigma))){
    stop("* ADMM.SPCA : input 'Sigma' is invalid data matrix.")  }
  p = nrow(Sigma)
  #   2. numpc
  numpc = as.integer(numpc)
  if ((numpc<1)||(numpc>=p)||(is.na(numpc))||(is.infinite(numpc))||(!is.numeric(numpc))){
    stop("* ADMM.SPCA : 'numpc' should be an integer in [1,nrow(Sigma)).")
  }
  #   3. mu, rho, abstol, reltol, maxiter
  mu  = as.double(mu)
  rho = as.double(rho)
  if (!check_param_constant(rho,0)){
    stop("* ADMM.SPCA : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu,0)){
    stop("* ADMM.SPCA : 'mu' should be a positive real number.")
  }
  if (!check_param_constant_multiple(c(abstol, reltol))){
    stop("* ADMM.SPCA : tolerance level is invalid.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.SPCA : 'maxiter' should be a positive integer.")
  }

  # -----------------------------------------------------------------
  ## MAIN ITERATION
  basis = array(0,c(p,numpc))
  history = list()
  for (i in 1:numpc){
    # 1. run cpp part
    runcpp = admm_spca(Sigma, reltol, abstol, maxiter, mu, rho)
    # 2. separate outputs
    tmpX    = runcpp$X
    tmpk    = runcpp$k
    tmphist = data.frame(r_norm=runcpp$r_norm[1:tmpk],
                         s_norm=runcpp$s_norm[1:tmpk],
                         eps_pri=runcpp$eps_pri[1:tmpk],
                         eps_dual=runcpp$eps_dual[1:tmpk])
    history[[i]] = tmphist
    # 3. rank-1 vector extraction
    solvec    = admm_spca_rk1vec(tmpX)
    basis[,i] = solvec
    # 4. update
    Sigma = admm_spca_deflation(Sigma, solvec)
  }

  # -----------------------------------------------------------------
  ## RETURN OUTPUT
  output = list()
  output$basis = basis
  output$history = history
  return(output)
}




# Schur complement deflation ----------------------------------------------
#' @keywords internal
#' @noRd
admm_spca_deflation <- function(Sig, vec){
  p = length(vec)
  term1 = (Sig%*%outer(vec,vec)%*%Sig)
  term2 = sum((as.vector(Sig%*%matrix(vec,nrow=p)))*vec)

  output = Sig - term1/term2
  return(output)
}


# Rank-1 extraction -------------------------------------------------------
#' @keywords internal
#' @noRd
admm_spca_rk1vec <- function(X){
  y = as.vector(base::eigen(X)$vectors[,1])
  return(y)
}
