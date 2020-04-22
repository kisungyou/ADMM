#' Elastic Net Regularization
#'
#' Elastic Net regularization is a combination of \eqn{\ell_2} stability and
#' \eqn{\ell_1} sparsity constraint simulatenously solving the following,
#' \deqn{\textrm{min}_x ~ \frac{1}{2}\|Ax-b\|_2^2 + \lambda_1 \|x\|_1 + \lambda_2 \|x\|_2^2}
#' with nonnegative constraints \eqn{\lambda_1} and \eqn{\lambda_2}. Note that if both lambda values are 0,
#' it reduces to least-squares solution.
#'
#' @param A an \eqn{(m\times n)} regressor matrix
#' @param b a length-\eqn{m} response vector
#' @param lambda1 a regularization parameter for \eqn{\ell_1} term
#' @param lambda2 a regularization parameter for \eqn{\ell_2} term
#' @param rho an augmented Lagrangian parameter
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
#' ## generate underdetermined design matrix
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
#' ## run example with both regularization values = 1
#' output = admm.enet(A, b, lambda1=1, lambda2=1)
#' niter  = length(output$history$s_norm)
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
#' \insertRef{zou_regularization_2005a}{ADMM}
#'
#' @seealso \code{\link{admm.lasso}}
#' @author Xiaozhi Zhu
#' @export
admm.enet <- function(A, b,
                      lambda1=1.0, lambda2=1.0, rho=1.0,
                      abstol=1e-4, reltol=1e-2, maxiter=1000){
  #-----------------------------------------------------------
  ## PREPROCESSING
  # data validity
  if (!check_data_matrix(A)){
    stop("* ADMM.ENET : input 'A' is invalid data matrix.")  }
  if (!check_data_vector(b)){
    stop("* ADMM.ENET : input 'b' is invalid data vector")  }
  b = as.vector(b)
  # data size
  if (nrow(A)!=length(b)){
    stop("* ADMM.ENET : two inputs 'A' and 'b' have non-matching dimension.")}
  # other parameters
  if (!check_param_constant_multiple(c(abstol, reltol))){
    stop("* ADMM.ENET : tolerance level is invalid.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.ENET : 'maxiter' should be a positive integer.")
  }
  maxiter = as.integer(maxiter)
  rho = as.double(rho)
  if (!check_param_constant(rho,0)){
    stop("* ADMM.ENET : 'rho' should be a positive real number.")
  }
  # adjust for Xiaozhi's code
  meps = (.Machine$double.eps)
  negsmall = -meps
  lambda1 = as.double(lambda1)
  lambda2 = as.double(lambda2)
  if (!check_param_constant(lambda1, negsmall)){
    stop("* ADMM.ENET : 'lambda1' is invalid.")
  }
  if (!check_param_constant(lambda2, negsmall)){
    stop("* ADMM.ENET : 'lambda2' is invalid.")
  }
  if ((lambda1<meps)&&(lambda2<meps)){
    message("* ADMM.ENET : since both regularization parameters are effectively zero, a least-squares solution is returned.")
    xsol   = as.vector(aux_pinv(A)%*%matrix(b))
    output = list()
    output$x = xsol
    return(output)
  }
  lambda = (2*lambda2 + lambda1)
  alpha  = (lambda1/lambda)

  #-----------------------------------------------------------
  ## MAIN COMPUTATION
  result = admm_enet(A,b,lambda,alpha,reltol,abstol,maxiter,rho)

  #-----------------------------------------------------------
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
