#' Least Absolute Shrinkage and Selection Operator
#'
#' LASSO, or L1-regularized regression, is an optimization problem to solve
#' \deqn{\textrm{min}_x ~ \frac{1}{2}\|Ax-b\|_2^2 + \lambda \|x\|_1}
#' for sparsifying the coefficient vector \eqn{x}.
#' The implementation is borrowed from Stephen Boyd's
#' \href{https://stanford.edu/~boyd/papers/admm/lasso/lasso.html}{MATLAB code}.
#'
#' @param A an \eqn{(m\times n)} regressor matrix
#' @param b a length-\eqn{m} response vector
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
#' @examples
#' \donttest{
#' ## generate sample data
#' m = 50
#' n = 100
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
#' lambda = 0.1*base::norm(t(A)%*%b, "F")
#'
#' ## run example
#' output  = admm.lasso(A, b, lambda)
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
#' }
#'
#' @references
#' \insertRef{tibshirani_regression_1996a}{ADMM}
#'
#' @export
admm.lasso <- function(A, b, lambda=1.0, rho=1.0, alpha=1.0,
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
  xinit = as.vector(rnorm(ncol(A))/10)
  # if (!is.na(xinit)){
  #   if ((!check_data_vector(xinit))||(length(xinit)!=ncol(A))){
  #     stop("* ADMM.LASSO : input 'xinit' is invalid.")
  #   }
  #   xinit = as.vector(xinit)
  # } else {
  #   xinit = as.vector(rep(0,ncol(A)))
  # }


  # other parameters
  meps = (.Machine$double.eps)
  negsmall = -meps
  if (!check_param_constant(lambda,negsmall)){
    stop("* ADMM.LASSO : reg. parameter 'lambda' is invalid.")
  }
  if (lambda < meps){
    message("* ADMM.LASSO : since 'lambda' is effectively zero, a least-squares solution is returned.")
    xsol   = as.vector(aux_pinv(A)%*%matrix(b))
    output = list()
    output$x = xsol
    return(output)
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

#
# lcost <- function(x){diff = as.vector(A%*%x-b); return((sum(diff*diff)/2)+lambda*sum(abs(x)))}

#
# 1. https://stackoverflow.com/questions/2247111/evaluating-variable-within-r-loop
# multiply <- function(i) {
#   force(i)
#   function(x) x * i
# }
# funcs <- list()
# for(i in 1:21){
#   funcName <- paste( 'func', i, sep = '' )
#   funcs[[funcName]] = multiply(i)
# }
#
# 2. https://stackoverflow.com/questions/15627701/r-scope-force-variable-substitution-in-function-without-local-environment?noredirect=1&lq=1
#
# 3. https://stackoverflow.com/questions/32100372/calling-functions-from-a-list-recursively?noredirect=1&lq=1
