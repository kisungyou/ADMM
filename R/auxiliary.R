# CHECKERS ----------------------------------------------------------------
#' @keywords internal
#' @noRd
check_data_matrix <- function(A){
  cond1 = (is.matrix(A))  # matrix
  cond2 = (!(any(is.infinite(A))||any(is.na(A))))
  if (cond1&&cond2){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

#' @keywords internal
#' @noRd
check_data_vector <- function(b){
  cond1 = ((is.vector(b))||((is.matrix(b))&&
    (length(b)==nrow(b))||(length(b)==ncol(b))))
  cond2 = (!(any(is.infinite(b))||any(is.na(b))))
  if (cond1&&cond2){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

#' @keywords internal
#' @noRd
check_param_constant <- function(num, lowerbound=0){
  cond1 = (length(num)==1)
  cond2 = ((!is.infinite(num))&&(!is.na(num)))
  cond3 = (num > lowerbound)

  if (cond1&&cond2&&cond3){
    return(TRUE)
  } else {
    return(FALSE)
  }
}

#' @keywords internal
#' @noRd
check_param_constant_multiple <- function(numvec, lowerbound=0){
  for (i in 1:length(numvec)){
    if (!check_param_constant(numvec[i], lowerbound)){
      return(FALSE)
    }
  }
  return(TRUE)
}


#' @keywords internal
#' @noRd
check_param_integer <- function(num, lowerbound=0){
  cond1 = (length(num)==1)
  cond2 = ((!is.infinite(num))&&(!is.na(num)))
  cond3 = (num > lowerbound)
  cond4 = (abs(num-round(num)) < sqrt(.Machine$double.eps))

  if (cond1&&cond2&&cond3&&cond4){
    return(TRUE)
  } else {
    return(FALSE)
  }
}



# auxiliary computations --------------------------------------------------
# 1. Regularized LU decomposition
#' @keywords internal
#' @noRd
boyd_factor <- function(A, rho){
  m = nrow(A)
  n = ncol(A)

  if (m>=n){ # if skinny matrix
    U = (chol(t(A)%*%A + rho*diag(n)))
  } else {
    U = (chol(diag(m)+(1/rho)*(A%*%t(A))))
  }
  output = list()
  output$L = t(U)
  output$U = U
}
# 2. PseudoInverse using SVD and NumPy Scheme
# https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Singular_value_decomposition_(SVD)
#' @keywords internal
#' @noRd
aux_pinv <- function(A){
  svdA      = base::svd(A)
  tolerance = (.Machine$double.eps)*max(c(nrow(A),ncol(A)))*as.double(max(svdA$d))

  idxcut    = which(svdA$d <= tolerance)
  invDvec   = (1/svdA$d)
  invDvec[idxcut] = 0

  output = (svdA$v%*%diag(invDvec)%*%t(svdA$u))
  return(output)
}
