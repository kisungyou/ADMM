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



# AUXILIARY COMPUTATIONS --------------------------------------------------
#   -----------------------------------------------------------------------
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
#   -----------------------------------------------------------------------
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
#   -----------------------------------------------------------------------
# 3. updatd for genenet
# inversion : use half of the cores
#' @keywords internal
#' @noRd
aux_genetinversion <- function(A,rho,L,R,lambda2,parallel=FALSE,nCore=ceiling(detectCores()/2)){
  # -----------------------------------------------------------------------
  # for the checker part, I pass, only need to

  # -----------------------------------------------------------------------
  # case 1 : lambda2 is a single value
  if (length(lambda2)==1){
    mat = aux_pinv((t(A)%*%A)+rho*(t(L)%*%L)+2*lambda2*(t(R)%*%R))
    return(mat)
  } else {
    # -----------------------------------------------------------------------
    # case 2 : lambda2 is a vector of lambda values
    if (parallel==TRUE){
      nCore   = max(1, as.integer(nCore))
      nlambda = length(lambda2)
      p       = ncol(A)

      cl = makeCluster(nCore)
      registerDoParallel(cl)

      iteach = NULL
      output = foreach (iteach=1:nlambda, .combine = cbind) %dopar% {
        aux_pinv((t(A)%*%A)+rho*(t(L)%*%L)+2*lambda2[iteach]*(t(R)%*%R))
      }
      stopCluster(cl)

      dim(output) = c(p,p,nlambda)
      return(output)
    } else { # this is CPP part
      output = multipleinversion(A,rho,L,R,lambda2)
      return(output)
    }
  }
}

# test1  : parallel (of R) : 1830803 vs 3986 : Sequential is Better
# microbenchmark("job1"={output1=aux_genetinversion(A,rho,L,R,lambda2,parallel=TRUE,nCore=1)},
#                "job2"={output2=aux_genetinversion(A,rho,L,R,lambda2,parallel=FALSE)}, times=10)
# test2  : number of Cores : not necessarily better to use more cores
#
# result : simply use RCPP VERSION : it is much faster


#   -----------------------------------------------------------------------
# 4. Laplacian L to R matrix : L = R^T * R
#' @keywords internal
#' @noRd
aux_laplacian2R <- function(L,size="auto"){
  if (!isSymmetric(L)){
    stop("we need symmetric matrix anyway.")
  }
  # possibly the rank
  rL = as.integer(Matrix::rankMatrix(L))
  if (size=="auto"){
    size = rL
  } else {
    size = as.integer(size)
    if (size>rL){
      message("* laplacian... hmm... auto adjust!")
      size=rL
    }
  }

  # use top eigenpairs
  eigL  = base::eigen(L)
  V     = eigL$vectors[,1:size]
  # for (i in 1:size){
  #   vecV  = as.vector(V[,i])
  #   normV = sqrt(sum(vecV*vecV))
  #   V[,i] = V[,i]/normV
  # }
  Dhalf = as.vector(sqrt(eigL$values[1:size]))
  R     = (diag(Dhalf)%*%t(V))
  return(R)
}
