#' Basis Pursuit
#'
#' For an underdetermined system, Basis Pursuit
#' aims to find a sparse solution that solves
#' \deqn{\textrm{min}_x ~  \|x\|_1 \quad \textrm{s.t} \quad Ax=b}
#' which is a relaxed version of strict non-zero support finding problem.
#' The implementation is borrowed from Stephen Boyd's
#' \href{https://web.stanford.edu/~boyd/papers/admm/basis_pursuit/basis_pursuit.html}{MATLAB code}.
#'
#' @param A an \eqn{(m \times n)} regressor matrix
#' @param b a length-\eqn{m} response vector
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
#' n = 30
#' m = 10
#'
#' A = matrix(rnorm(n*m), nrow=m)    # design matrix
#' x = c(stats::rnorm(3),rep(0,n-3)) # coefficient
#' x = base::sample(x)
#' b = as.vector(A%*%x)              # response
#'
#' ## run example
#' output  = admm.bp(A, b)
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
#' @export
admm.bp <- function(A, b, xinit=NA,
                    rho=1.0, alpha=1.0,
                    abstol=1e-4, reltol=1e-2, maxiter=1000){
  ## PREPROCESSING
  # data validity
  if (!check_data_matrix(A)){
    stop("* ADMM.BP : input 'A' is invalid data matrix.")  }
  if (!check_data_vector(b)){
    stop("* ADMM.BP : input 'b' is invalid data vector")  }
  b = as.vector(b)
  # data size
  if (nrow(A)!=length(b)){
    stop("* ADMM.BP : two inputs 'A' and 'b' have non-matching dimension.")}
  # initial value
  if (!is.na(xinit)){
    if ((!check_data_vector(xinit))||(length(xinit)!=ncol(A))){
      stop("* ADMM.BP : input 'xinit' is invalid.")
    }
    xinit = as.vector(xinit)
  } else {
    xinit = as.vector(rep(0,ncol(A)))
  }
  # other parameters
  if (!check_param_constant_multiple(c(abstol, reltol))){
    stop("* ADMM.BP : tolerance level is invalid.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.BP : 'maxiter' should be a positive integer.")
  }
  maxiter = as.integer(maxiter)
  if (!check_param_constant(rho,0)){
    stop("* ADMM.BP : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(alpha,0)){
    stop("* ADMM.BP : 'alpha' should be a positive real number.")
  }
  if ((alpha<1)||(alpha>2)){
    warning("* ADMM.BP : 'alpha' value is suggested to be in [1,2].")
  }

  ## MAIN COMPUTATION & RESULT RETURN
  result = admm_bp(A,b,xinit,reltol,abstol,maxiter,rho,alpha)

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
