__glm__ estimates [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs) in Go.

See the [examples](http://github.com/kshedden/statmodel/tree/master/glm/examples)
directory for examples.  This package can be used to produce results such as
[this](http://github.com/kshedden/statmodel/tree/master/glm/examples/nhanes/nhanes.md).

Supported features
------------------

* Estimation via IRLS and [gonum](http://github.com/gonum) optimizers

* Supports many GLM families, links and variance functions

* Supports estimation for case-weighted datasets

* Models can be specified using formulas

* Regularized (ridge/LASSO/elastic net) estimation

* Offsets

* Unit tests covering all families with their default links and
  variance functions, and some of the more common non-canonical links


Missing features
----------------

* Performance assessments

* Model diagnostics

* Marginalization

* Missing data handling

* GEE

* Inference for survey data
