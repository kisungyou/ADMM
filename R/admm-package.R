#' ADMM : Algorithms using Alternating Direction Method of Multipliers
#'
#' An introduction of Alternating Direction Method of Multipliers (ADMM) method has been a breakthrough in
#' solving complex and non-convex optimization problems in a reasonably stable as well as scalable fashion.
#' Our package aims at providing handy tools for fast computation on well-known problems using the method.
#' For interested users/readers, please visit Prof. Stephen Boyd's \href{https://stanford.edu/~boyd/papers/admm_distr_stats.html}{website}
#' entirely devoted to the topic.
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
#' @importFrom utils packageVersion
#' @useDynLib ADMM
NULL


