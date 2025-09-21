#' ADMM : Algorithms using Alternating Direction Method of Multipliers
#'
#' @noRd
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


