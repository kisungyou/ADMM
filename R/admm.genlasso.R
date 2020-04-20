#' Generalized LASSO
#'
#' Generalized LASSO is solving the following equation,
#' \deqn{\textrm{min}_x ~ \frac{1}{2}\|Ax-b\|_2^2 + \lambda \|Dx\|_1}
#' where the choice of regularization matrix \eqn{D} leads to different problem formulations.
#'
#' @param A an \eqn{(m\times n)} regressor matrix
#' @param b a length-\eqn{m} response vector
#' @param D a regularization matrix of \eqn{n} columns
#' @param lambda a regularization parameter
#' @param rho an augmented Lagrangian parameter
#' @param alpha an overrelaxation parameter in [1,2]
#' @param abstol absolute tolerance stopping criterion
#' @param reltol relative tolerance stopping criterion
#' @param maxiter maximum number of iterations
#'
#' @return a named list containing \describe{
#' \item{x}{a length-\eqn{n} solution vector}
#' \item{history}{dataframe recording iteration numerics. See the section for more details.}
#' }
#'
#' @section Iteration History:
#' When you run the algorithm, output returns not only the solution, but also the iteration history recording
#' following fields over iterates,
#' \describe{
#' \item{objval}{object (cost) function value}
#' \item{r_norm}{norm of primal residual}
#' \item{s_norm}{norm of dual residual}
#' \item{eps_pri}{feasibility tolerance for primal feasibility condition}
#' \item{eps_dual}{feasibility tolerance for dual feasibility condition}
#' }
#' In accordance with the paper, iteration stops when both \code{r_norm} and \code{s_norm} values
#' become smaller than \code{eps_pri} and \code{eps_dual}, respectively.
#'
#'
#' @examples
#' ## generate sample data
#' m = 100
#' n = 200
#' p = 0.1   # percentange of non-zero elements
#'
#' x0 = matrix(Matrix::rsparsematrix(n,1,p))
#' A  = matrix(rnorm(m*n),nrow=m)
#' for (i in 1:ncol(A)){
#'   A[,i] = A[,i]/sqrt(sum(A[,i]*A[,i]))
#' }
#' b = A%*%x0 + sqrt(0.001)*matrix(rnorm(m))
#' D = diag(n);
#'
#' ## set regularization lambda value
#' regval = 0.1*Matrix::norm(t(A)%*%b, 'I')
#'
#' ## solve LASSO via reducing from Generalized LASSO
#' output  = admm.genlasso(A,b,D,lambda=regval) # set D as identity matrix
#' niter   = length(output$history$s_norm)
#' history = output$history
#'
#' ## report convergence plot
#' opar <- par(no.readonly=TRUE)
#' par(mfrow=c(1,3))
#' plot(1:niter, history$objval, "b", main="cost function")
#' plot(1:niter, history$r_norm, "b", main="primal residual")
#' plot(1:niter, history$s_norm, "b", main="dual residual")
#' par(opar)
#'
#' @references
#' \insertRef{tibshirani_solution_2011}{ADMM}
#'
#' \insertRef{zhu_augmented_2017}{ADMM}
#'
#' @author Xiaozhi Zhu
#' @export
admm.genlasso <- function(A, b, D=diag(length(b)), lambda=1.0, rho=1.0, alpha=1.0,
                       abstol=1e-4, reltol=1e-2, maxiter=1000){
  #-----------------------------------------------------------
  ## PREPROCESSING
  # 1. data validity
  if (!check_data_matrix(A)){
    stop("* ADMM.GENLASSO : input 'A' is invalid data matrix.")  }
  if (!check_data_vector(b)){
    stop("* ADMM.GENLASSO : input 'b' is invalid data vector")  }
  b = as.vector(b)
  # 2. data size
  if (nrow(A)!=length(b)){
    stop("* ADMM.GENLASSO : two inputs 'A' and 'b' have non-matching dimension.")}
  # 3. D : regularization matrix
  if (!check_data_matrix(D)){
    stop("* ADMM.GENLASSO : input 'D' is invalid regularization matrix.")
  }
  if (ncol(A)!=ncol(D)){
    stop("* ADMM.GENLASSO : input 'D' has invalid size.")
  }
  # 4. other parameters
  if (!check_param_constant_multiple(c(abstol, reltol))){
    stop("* ADMM.GENLASSO : tolerance level is invalid.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.GENLASSO : 'maxiter' should be a positive integer.")
  }
  maxiter = as.integer(maxiter)
  rho = as.double(rho)
  if (!check_param_constant(rho,0)){
    stop("* ADMM.GENLASSO : 'rho' should be a positive real number.")
  }

  #-----------------------------------------------------------
  ## MAIN COMPUTATION
  #   1. lambda=0 case; pseudoinverse
  meps     = (.Machine$double.eps)
  negsmall = -meps
  lambda   = as.double(lambda)
  if (!check_param_constant(lambda, negsmall)){
    stop("* ADMM.GENLASSO : 'lambda' is invalid; should be a nonnegative real number.")
  }
  if (lambda<meps){
    message("* ADMM.GENLASSO : since both regularization parameters are effectively zero, a least-squares solution is returned.")
    xsol   = as.vector(aux_pinv(A)%*%matrix(b))
    output = list()
    output$x = xsol
    return(output)
  }

  #   2. main computation : Xiaozhi's work
  result = admm_genlasso(A,b,D,lambda,reltol,abstol,maxiter,rho)


  #-----------------------------------------------------------
  ## RESULT RETURN
  kk = result$k
  output = list()
  output$x = result$x
  output$history = data.frame(objval=result$objval[1:kk],
                              r_norm=result$r_norm[1:kk],
                              s_norm=result$s_norm[1:kk],
                              eps_pri=result$eps_pri[1:kk],
                              eps_dual=result$eps_dual[1:kk])
  return(output)
}




