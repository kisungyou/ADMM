#' ADMM : Algorithms using Alternating Direction Method of Multipliers
#'
#' An introduction of Alternating Direction Method of Multipliers (ADMM) method has been a breakthrough in
#' solving complex and non-convex optimization problems in a reasonably stable as well as scalable fashion.
#' Our package aims at providing handy tools for fast computation on well-known problems using the method.
#' For interested users/readers, please visit Prof. Stephen Boyd's \href{http://stanford.edu/~boyd/papers/admm_distr_stats.html}{website}
#' entirely devoted to the topic. Below is the list of functions supported,
#' \tabular{cl}{
#' FUNCTION \tab Algorithm \cr
#' \code{\link{admm.bp}} \tab Basis Pursuit \cr
#' \code{\link{admm.enet}} \tab Elastic Net Regularization \cr
#' \code{\link{admm.genlasso}} \tab Generalized LASSO \cr
#' \code{\link{admm.lad}} \tab Least Absolute Deviations \cr
#' \code{\link{admm.lasso}} \tab Least Absolute Shrinkage and Selection Operator \cr
#' \code{\link{admm.rpca}} \tab Robust Principal Component Analysis \cr
#' \code{\link{admm.spca}} \tab Sparse Principal Component Analysis \cr
#' \code{\link{admm.tv}} \tab Total Variation Minimization
#' }
#'
#' @docType package
#' @name ADMM
#' @aliases ADMM-package
#' @import Rdpack
#' @import Matrix
#' @importFrom Rcpp evalCpp
#' @importFrom stats rnorm
#' @importFrom foreach "%dopar%" foreach registerDoSEQ
#' @importFrom parallel detectCores stopCluster makeCluster
#' @importFrom doParallel registerDoParallel
#' @useDynLib ADMM
NULL


