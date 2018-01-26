#' LASSO
#'
#' LASSO, or L1-regularized regression, is an optimization problem to solve
#' \deqn{min_x ~ \frac{1}{2}\|Ax-b\|_2^2 + \lambda \|x\|_1}
#' for sparsifying the coefficient vector \eqn{x}.
#' The implementation is borrowed from Stephen Boyd's
#' \href{http://stanford.edu/~boyd/papers/admm/lasso/lasso.html}{MATLAB code}.
#'
#' @param A an \eqn{(m\times n)} regressor matrix
#' @param b a length-\eqn{m} response vector
#' @param lambda a regularization parameber
#' @param xinit a length-\eqn{n} vector for initial value
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
#' @examples
#' ## generate sample data
#' m = 500
#' n = 1000
#' p = 0.1   # percentange of non-zero elements
#'
#' x0 = matrix(Matrix::rsparsematrix(n,1,p))
#' A  = matrix(rnorm(m*n),nrow=m)
#' for (i in 1:ncol(A)){
#'   A[,i] = A[,i]/sqrt(sum(A[,i]*A[,i]))
#' }
#' b = A%*%x0 + sqrt(0.001)*matrix(rnorm(m))
#'
#' ## set regularization lambda value
#' lambda = 0.1*Matrix::norm(t(A)%*%b, 'I')
#'
#' ## run example
#' output = admm.lasso(A, b, lambda)
#'
#' ## report convergence plot
#' niter  = length(output$history$s_norm)
#' par(mfrow=c(1,3))
#' plot(1:niter, output$history$objval, "b", main="cost function")
#' plot(1:niter, output$history$r_norm, "b", main="primal residual")
#' plot(1:niter, output$history$s_norm, "b", main="dual residual")
#'
#' @rdname LASSO
#' @export
admm.lasso <- function(A, b, lambda=1.0, xinit=NA,
                    rho=1.0, alpha=1.0,
                    abstol=1e-4, reltol=1e-2, maxiter=1000){
  ## PREPROCESSING
  # data validity
  if (!check_data_matrix(A)){
    stop("* ADMM.LASSO : input 'A' is invalid data matrix.")  }
  if (!check_data_vector(b)){
    stop("* ADMM.LASSO : input 'b' is invalid data vector")  }
  b = as.vector(b)
  # data size
  if (nrow(A)!=length(b)){
    stop("* ADMM.LASSO : two inputs 'A' and 'b' have non-matching dimension.")}
  # initial value
  if (!is.na(xinit)){
    if ((!check_data_vector(xinit))||(length(xinit)!=ncol(A))){
      stop("* ADMM.LASSO : input 'xinit' is invalid.")
    }
    xinit = as.vector(xinit)
  } else {
    xinit = as.vector(rep(0,ncol(A)))
  }
  # other parameters
  if (!check_param_constant(lambda)){
    stop("* ADMM.LASSO : reg. parameter 'lambda' is invalid.")
  }
  if (!check_param_constant_multiple(c(abstol, reltol))){
    stop("* ADMM.LASSO : tolerance level is invalid.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.LASSO : 'maxiter' should be a positive integer.")
  }
  maxiter = as.integer(maxiter)
  if (!check_param_constant(rho,0)){
    stop("* ADMM.LASSO : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(alpha,0)){
    stop("* ADMM.LASSO : 'alpha' should be a positive real number.")
  }
  if ((alpha<1)||(alpha>2)){
    warning("* ADMM.LASSO : 'alpha' value is suggested to be in [1,2].")
  }

  ## MAIN COMPUTATION & RESULT RETURN
  result = admm_lasso(A,b,lambda,xinit,reltol,abstol,maxiter,rho,alpha)

  ## RESULT RETURN
  kk = result$k
  output = list()
  output$x = result$x
  output$history = data.frame(objval=result$objval[1:kk],
                              r_norm=result$r_norm[1:kk],
                              s_norm=result$s_norm[1:kk],
                              eps_pri=result$eps_pri[1:kk],
                              eps_dual=result$eps_dual[1:kk]
  )
  return(output)
}