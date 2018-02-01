#' Sparse PCA
#'
#'
#'
#' @rdname SPCA
#' @export
admm.spca <- function(Sigma, numpc, mu=1.0, rho=1.0, abstol=1e-4, reltol=1e-2, maxiter=1000){
  # -----------------------------------------------------------------
  ## PREPROCESSING
  #   1. data
  if ((!check_data_matrix(Sigma))||(!isSymmetric(Sigma))){
    stop("* ADMM.SPCA : input 'Sigma' is invalid data matrix.")  }
  p = nrow(Sigma)
  #   2. numpc
  numpc = as.integer(numpc)
  if ((numpc<1)||(numpc>=p)||(is.na(numpc))||(is.infinite(numpc))||(!is.numeric(numpc))){
    stop("* ADMM.SPCA : 'numpc' should be an integer in [1,nrow(Sigma)).")
  }
  #   3. mu, rho, abstol, reltol, maxiter
  mu  = as.double(mu)
  rho = as.double(rho)
  if (!check_param_constant(rho,0)){
    stop("* ADMM.SPCA : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu,0)){
    stop("* ADMM.SPCA : 'mu' should be a positive real number.")
  }
  if (!check_param_constant_multiple(c(abstol, reltol))){
    stop("* ADMM.SPCA : tolerance level is invalid.")
  }
  if (!check_param_integer(maxiter, 2)){
    stop("* ADMM.SPCA : 'maxiter' should be a positive integer.")
  }

  # -----------------------------------------------------------------
  ## MAIN ITERATION
  basis = array(0,c(p,numpc))
  history = list()
  for (i in 1:numpc){
    # 1. run cpp part
    runcpp = admm_spca(Sigma, reltol, abstol, maxiter, mu, rho)
    # 2. separate outputs
    tmpX    = runcpp$X
    tmpk    = runcpp$k
    tmphist = data.frame(r_norm=runcpp$r_norm[1:tmpk],
                         s_norm=runcpp$s_norm[1:tmpk],
                         eps_pri=runcpp$eps_pri[1:tmpk],
                         eps_dual=runcpp$eps_dual[1:tmpk])
    history[[i]] = tmphist
    # 3. rank-1 vector extraction
    solvec    = admm_spca_rk1vec(tmpX)
    basis[,i] = solvec
    # 4. update
    Sigma = admm_spca_deflation(Sigma, solvec)
  }

  # -----------------------------------------------------------------
  ## RETURN OUTPUT
  output = list()
  output$basis = basis
  output$history = history
  return(output)
}




# Schur complement deflation ----------------------------------------------
#' @keywords internal
#' @noRd
admm_spca_deflation <- function(Sig, vec){
  p = length(vec)
  term1 = (Sig%*%outer(vec,vec)%*%Sig)
  term2 = sum((as.vector(Sig%*%matrix(vec,nrow=p)))*vec)

  output = Sig - term1/term2
  return(output)
}


# Rank-1 extraction -------------------------------------------------------
#' @keywords internal
#' @noRd
admm_spca_rk1vec <- function(X){
  y = as.vector(base::eigen(X)$vectors[,1])
  return(y)
}
