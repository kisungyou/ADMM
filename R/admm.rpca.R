#' Robust Principal Component Analysis
#'
#' Given a data matrix \eqn{M}, it finds a decomposition
#' \deqn{\textrm{min}~\|L\|_*+\lambda \|S\|_1\quad \textrm{s.t.}\quad L+S=M}
#' where \eqn{\|L\|_*} represents a nuclear norm for a matrix \eqn{L} and
#' \eqn{\|S\|_1 = \sum |S_{i,j}|}, and \eqn{\lambda} a balancing/regularization
#' parameter. The choice of such norms leads to impose \emph{low-rank} property for \eqn{L} and
#' \emph{sparsity} on \eqn{S}.
#'
#'
#' @param M an \eqn{(m\times n)} data matrix
#' @param lambda a regularization parameter
#' @param mu an augmented Lagrangian parameter
#' @param tol relative tolerance stopping criterion
#' @param maxiter maximum number of iterations
#'
#' @return a named list containing \describe{
#' \item{L}{an \eqn{(m\times n)} low-rank matrix}
#' \item{S}{an \eqn{(m\times n)} sparse matrix}
#' \item{history}{dataframe recording iteration numerics. See the section for more details.}
#' }
#'
#' @section Iteration History:
#' For RPCA implementation, we chose a very simple stopping criterion
#' \deqn{\|M-(L_k+S_k)\|_F \le tol*\|M\|_F}
#' for each iteration step \eqn{k}. So for this method, we provide a vector of only relative errors,
#' \describe{
#' \item{error}{relative error computed}
#' }
#'
#' @examples
#' ## generate data matrix from standard normal
#' X = matrix(rnorm(20*5),nrow=5)
#'
#' ## try different regularization values
#' out1 = admm.rpca(X, lambda=0.01)
#' out2 = admm.rpca(X, lambda=0.1)
#' out3 = admm.rpca(X, lambda=1)
#'
#' ## visualize sparsity
#' opar <- par(no.readonly=TRUE)
#' par(mfrow=c(1,3))
#' image(out1$S, main="lambda=0.01")
#' image(out2$S, main="lambda=0.1")
#' image(out3$S, main="lambda=1")
#' par(opar)
#'
#' @references
#' \insertRef{candes_robust_2011a}{ADMM}
#'
#' @export
admm.rpca <- function(M, lambda=1/sqrt(max(nrow(M),ncol(M))), mu=1.0, tol=1e-7, maxiter=1000){
  # -----------------------------------------------------------------
  ## PREPROCESSING
  #   1. data M
  if (!check_data_matrix(M)){
    stop("* ADMM.RPCA : input 'M' is invalid data matrix.")
  }
  #   2. lambda and mu
  lambda = as.double(lambda)
  mu     = as.double(mu)
  if (!check_param_constant(mu,0)){
    stop("* ADMM.RPCA : 'mu' should be a positive real number.")
  }
  if (!check_param_constant(lambda,0)){
    stop("* ADMM.RPCA : 'lambda' should be a positive real number.")
  }
  #   3. tol and maxiter
  tol     = as.double(tol)
  maxiter = as.integer(maxiter)
  if (!check_param_constant(tol,0)){
    stop("* ADMM.RPCA : 'tol' should be a positive real number.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.RPCA : 'maxiter' should be a positive integer.")
  }

  # -----------------------------------------------------------------
  ## MAIN ITERATION
  #   1. run CPP computation
  runcpp = admm_rpca(M, tol, maxiter, mu, lambda)
  #   2. separate out the results
  kk = runcpp$k

  # -----------------------------------------------------------------
  ## RETURN THE OUTPUT
  output = list()
  output$L = runcpp$L
  output$S = runcpp$S
  output$history = data.frame(error=runcpp$errors[1:kk])
  return(output)
}
