__glm__ supports estimation of generalized linear models in Go.

The Godoc package documentation is [here](https://godoc.org/github.com/kshedden/statmodel/glm).

See the [examples](http://github.com/kshedden/statmodel/tree/master/glm/examples)
directory for examples, producing results such as
[this](http://github.com/kshedden/statmodel/tree/master/glm/examples/nhanes.md).

A basic usage example is as follows:

```
fam := glm.NewFamily(glm.BinomialFamily)
// data is a dstream
glm := glm.NewGLM(data, "Y").Family(fam).Done()
rslt := glm.Fit()
print(rslt.Summary().String())
```

`NewFamily` returns a GLM family (e.g. `Binomial`), and `data` is a
"Dstream" as defined in the
[dstream](http://github.com/kshedden/dstream)
package.  The Dstream is used to feed data to the GLM in chunks
using a column-oriented storage layout.  A more extensive illustration
can be found in the "examples" directory.


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

* Less-common GLM families (e.g. Tweedie)

* Marginalization

* Missing data handling

* GEE

* Inference for survey data
