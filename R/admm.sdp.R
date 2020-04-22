#' Semidefinite Programming
#'
#' We solve the following standard semidefinite programming (SDP) problem
#' \deqn{\textrm{min}_X ~ \textrm{tr}(CX)}
#' \deqn{\textrm{s.t.} A(X)=b, ~ X \geq 0 }
#' with \eqn{A(X)_i = \textrm{tr}(A_i^\top X) = b_i} for \eqn{i=1,\ldots,m} and \eqn{X \geq 0} stands for positive-definiteness of the matrix \eqn{X}. In the standard form,
#' matrices \eqn{C, A_1,A_2,\ldots,A_m} are symmetric and solution \eqn{X} would be symmetric and positive semidefinite. This function implements alternating direction augmented Lagrangian methods.
#'
#' @param C an \eqn{(n\times n)} symmetric matrix for cost.
#' @param A a length-\eqn{m} list of \eqn{(n\times n)} symmetric matrices for constraint.
#' @param b a length-\eqn{m} vector for equality condition.
#' @param mu penalty parameter; positive real number.
#' @param rho step size for updating in \eqn{(0, \frac{1+\sqrt{5}}{2})}.
#' @param abstol absolute tolerance stopping criterion.
#' @param maxiter maximum number of iterations.
#' @param print.progress a logical; \code{TRUE} to show the progress, \code{FALSE} to go silent.
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
#' \item{eps_pri}{feasibility tolerance for primal feasibility condition}
#' \item{eps_dual}{feasibility tolerance for dual feasibility condition}
#' \item{gap}{gap between primal and dual cost function.}
#' }
#' We use the stopping criterion which breaks the iteration when all \code{eps_pri},\code{eps_dual}, and \code{gap}
#' become smaller than \code{abstol}.
#'
#' @examples
#' ## a toy example
#' #  generate parameters
#' C  = matrix(c(1,2,3,2,9,0,3,0,7),nrow=3,byrow=TRUE)
#' A1 = matrix(c(1,0,1,0,3,7,1,7,5),nrow=3,byrow=TRUE)
#' A2 = matrix(c(0,2,8,2,6,0,8,0,4),nrow=3,byrow=TRUE)
#'
#' A  = list(A1, A2)
#' b  = c(11, 19)
#'
#' # run the algorithm
#' run = admm.sdp(C,A,b)
#' hst = run$history
#'
#' # visualize
#' opar <- par(no.readonly=TRUE)
#' par(mfrow=c(2,2))
#' plot(hst$objval,   type="b", cex=0.25, main="objective value")
#' plot(hst$eps_pri,  type="b", cex=0.25, main="primal feasibility")
#' plot(hst$eps_dual, type="b", cex=0.25, main="dual feasibility")
#' plot(hst$gap,      type="b", cex=0.25, main="primal-dual gap")
#' par(opar)
#'
#' \dontrun{
#' ## comparison with CVXR's result
#' require(CVXR)
#'
#' #  problems definition
#' X = Variable(3,3,PSD=TRUE)
#' myobj = Minimize(sum_entries(C*X)) # objective
#' mycon = list(                      # constraint
#'   sum_entries(A[[1]]*X) == b[1],
#'   sum_entries(A[[2]]*X) == b[2]
#' )
#' myp = Problem(myobj, mycon)        # problem
#'
#' # run and visualize
#' res  = solve(myp)
#' Xsol = res$getValue(X)
#'
#' opar = par(no.readonly=TRUE)
#' par(mfrow=c(1,2), pty="s")
#' image(run$X, axes=FALSE, main="ADMM result")
#' image(Xsol,  axes=FALSE, main="CVXR result")
#' par(opar)
#' }
#'
#' @references
#' \insertRef{wen_alternating_2010a}{ADMM}
#'
#' @author Kisung You
#' @export
admm.sdp <- function(C, A, b, mu=1.0, rho=1, abstol=1e-10, maxiter=496, print.progress=FALSE){
  #----------------------------------------------------------------------
  ## PREPROCESSING
  if ((!is.matrix(C))||(!base::isSymmetric(C))){
    stop("* admm.sdp : 'C' should be a symmetric matrix.")
  }
  n = base::nrow(C)
  if (!is.list(A)){
    stop("* admm.sdp : 'A' should be a list.")
  }
  m = length(A)
  if ((!is.vector(b))||(length(b)!=m)){
    stop("* admm.sdp : 'b' should be a vector having same length as 'A'.")
  }

  Anrow = unique(unlist(lapply(A, base::nrow)))
  Ancol = unique(unlist(lapply(A, base::ncol)))

  cond1 = all(unlist(lapply(A, base::isSymmetric))==TRUE)
  cond2 = FALSE
  cond3 = FALSE
  if (length(Anrow)==1){    if (Anrow==n){      cond2 = TRUE    }  }
  if (length(Ancol)==1){    if (Ancol==n){      cond3 = TRUE    }  }
  if (!(cond1&&cond2&&cond3)){
    stop("* admm.sdp : 'A' should be a list of symmetric matrices having same size as 'C'.")
  }
  mymu    = ifelse(mu > 0, as.double(mu), stop("* admm.sdp : 'mu' should be a positive real number."))
  myrho   = ifelse(((rho > 0)&&(rho < (1+sqrt(5))/2)), as.double(rho), stop("* admm.sdp : 'rho' should be a number in (0,(1+sqrt(5))/2)."))
  myiter  = round(maxiter)
  mytol   = as.double(abstol)

  mygamma = 0.9
  myprint = as.logical(print.progress)

  #----------------------------------------------------------------------
  ## Run
  result = admm_sdp(C, A, b, mymu, myrho, mygamma, myiter, mytol, myprint)

  #----------------------------------------------------------------------
  ## Wrap and Report
  output = list()
  output$X       = result$X
  output$history = data.frame(objval=result$objval,
                              eps_pri=result$eps_pri,
                              eps_dual=result$eps_dual,
                              gap=result$gap)
  return(output)
}


# pack <- "ADMM"
# path <- find.package(pack)
# system(paste(shQuote(file.path(R.home("bin"), "R")),
#              "CMD", "Rd2pdf", shQuote(path)))

# C  = matrix(c(1,2,3,2,9,0,3,0,7),nrow=3,byrow=TRUE)
# A1 = matrix(c(1,0,1,0,3,7,1,7,5),nrow=3,byrow=TRUE)
# A2 = matrix(c(0,2,8,2,6,0,8,0,4),nrow=3,byrow=TRUE)
#
# A  = list(A1, A2)
# b = c(11, 19)
#
# output = admm.sdp(C,A,b)
#
# library(CVXR)
#
# X = Variable(3,3,PSD=TRUE)
# myobj = Minimize(sum_entries(C*X))
# mycon = list(
#   sum_entries(A[[1]]*X) == b[1],
#   sum_entries(A[[2]]*X) == b[2]
# )
# myp = Problem(myobj, mycon)
# res = solve(myp)
# Xsol = res$getValue(X)
