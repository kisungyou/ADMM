
<!-- README.md is generated from README.Rmd. Please edit that file -->
ADMM
====

<!-- badges: start -->
[![CRAN status](https://www.r-pkg.org/badges/version/ADMM)](https://CRAN.R-project.org/package=ADMM) [![Travis build status](https://travis-ci.com/kyoustat/ADMM.svg?branch=master)](https://travis-ci.com/kyoustat/ADMM) <!-- badges: end -->

We provide implementation for a class of problems that use alternating direction method of multipliers (ADMM)-type algorithms.

Installation
------------

You can install the released version of ADMM from [CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("ADMM")
```

And the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("kyoustat/ADMM")
```

Available Functions
-------------------

Currently, we support following classes of problems and functions. For more details, please see help pages of each function using `help()` function in your R session.

|     Function    | Description                                     |
|:---------------:|:------------------------------------------------|
|    `admm.bp`    | Basis Pursuit                                   |
|   `admm.enet`   | Elastic Net Regularization                      |
| `admm.genlasso` | Generalized LASSO                               |
|    `admm.lad`   | Least Absolute Deviations                       |
|   `admm.lasso`  | Least Absolute Shrinkage and Selection Operator |
|   `admm.rpca`   | Robust Principal Component Analysis             |
|    `admm.sdp`   | Semidefinite Programming                        |
|   `admm.spca`   | Sparse Principal Component Analysis             |
|    `admm.tv`    | Total Variation Minimization                    |
