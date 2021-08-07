#' Total Variation Minimization
#'
#' 1-dimensional total variation minimization - also known as
#' signal denoising - is to solve the following
#' \deqn{\textrm{min}_x ~ \frac{1}{2}\|x-b\|_2^2 + \lambda \sum_i |x_{i+1}-x_i|}
#' for a given signal \eqn{b}.
#' The implementation is borrowed from Stephen Boyd's
#' \href{https://stanford.edu/~boyd/papers/admm/total_variation/total_variation.html}{MATLAB code}.
#'
#' @param b a length-\eqn{m} response vector
#' @param lambda regularization parameter
#' @param xinit a length-\eqn{m} vector for initial value
#' @param rho an augmented Lagrangian parameter
#' @param alpha an overrelaxation parameter in \eqn{[1,2]}
#' @param abstol absolute tolerance stopping criterion
#' @param reltol relative tolerance stopping criterion
#' @param maxiter maximum number of iterations
#'
#' @return a named list containing \describe{
#' \item{x}{a length-\eqn{m} solution vector}
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
#' x1 = as.vector(sin(1:100)+0.1*rnorm(100))
#' x2 = as.vector(cos(1:100)+0.1*rnorm(100)+5)
#' x3 = as.vector(sin(1:100)+0.1*rnorm(100)+2.5)
#' xsignal = c(x1,x2,x3)
#'
#' ## run example
#' output  = admm.tv(xsignal)
#'
#' ## visualize
#' opar <- par(no.readonly=TRUE)
#' plot(1:300, xsignal, type="l", main="TV Regularization")
#' lines(1:300, output$x, col="red", lwd=2)
#' par(opar)
#'
#' @export
admm.tv <- function(b, lambda=1.0, xinit=NA,
                    rho=1.0, alpha=1.0, abstol=1e-4, reltol=1e-2, maxiter=1000){
  ## PREPROCESSING
  # data validity
  if (!check_data_vector(b)){
    stop("* ADMM.TV : input 'b' is invalid data vector")  }
  b = as.vector(b)
  # data size
  # initial value
  if (!is.na(xinit)){
    if ((!check_data_vector(xinit))||(length(xinit)!=length(b))){
      stop("* ADMM.TV : input 'xinit' is invalid.")
    }
    xinit = as.vector(xinit)
  } else {
    xinit = as.vector(rep(0,length(b)))
  }
  # other parameters
  if (!check_param_constant_multiple(c(abstol, reltol))){
    stop("* ADMM.TV : tolerance level is invalid.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.TV : 'maxiter' should be a positive integer.")
  }
  maxiter = as.integer(maxiter)
  if (!check_param_constant(rho,0)){
    stop("* ADMM.TV : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(alpha,0)){
    stop("* ADMM.TV : 'alpha' should be a positive real number.")
  }
  if ((alpha<1)||(alpha>2)){
    warning("* ADMM.TV : 'alpha' value is suggested to be in [1,2].")
  }

  ## MAIN COMPUTATION & RESULT RETURN
  result = admm_tv(b, xinit, lambda, reltol, abstol, maxiter, rho, alpha)

  ## RESULT RETURN
  klength = result$k
  output = list()
  output$x = result$x
  output$history = data.frame(objval=result$objval[1:klength],
                              r_norm=result$r_norm[1:klength],
                              s_norm=result$s_norm[1:klength],
                              eps_pri=result$eps_pri[1:klength],
                              eps_dual=result$eps_dual[1:klength]
  )
  return(output)
}
