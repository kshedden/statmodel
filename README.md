[![Build Status](https://travis-ci.com/kshedden/statmodel.svg?branch=master)](https://travis-ci.com/kshedden/statmodel)
[![Go Report Card](https://goreportcard.com/badge/github.com/kshedden/statmodel)](https://goreportcard.com/report/github.com/kshedden/statmodel)
[![codecov](https://codecov.io/gh/kshedden/statmodel/branch/master/graph/badge.svg)](https://codecov.io/gh/kshedden/statmodel)
[![GoDoc](https://godoc.org/github.com/kshedden/statmodel?status.png)](https://godoc.org/github.com/kshedden/statmodel)

__statmodel__ is a collection of [Go](http://golang.org) packages for fitting
statistical models.

All results should agree to high precision with standard statistical packaged (R, Stata, SAS, etc.).  Extensive
unit tests against these packages are included in the test suite.

All models can be fit with maximum (or quasi-maximum) likelihood estimation, with optional L1 (Lasso) or L2 (ridge)
penalization.

The data provided to the fitting routines are stored column-wise (variable-wise) as either `[]float64` or
`[]float32` slices.  The data type is controlled by the `Dtype` parameter in the `statmodel/core.go` file.
Note that changing the data type involves recompiling the package.  All calculations are carried out in
`float64` precision, the option to provide the data as `float32` values is intended to improve cache performance,
and to enable analysis with very large data sets.

See the following packages for specific models:

* [glm](https://github.com/kshedden/statmodel/tree/master/glm) for Generalized Linear Models

* [duration](https://github.com/kshedden/statmodel/tree/master/duration) for survival analysis
