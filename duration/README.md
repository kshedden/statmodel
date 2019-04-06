The __duration__ package supports fitting statistical models for event
time data (also known as "survival analysis") in Go.  Currently, Cox
proportional hazards regression, Kaplan-Meier estimates of the
marginal survival function, and estimation of the cumulative incidence
function are supported.  The analysis functions accept data using the
[dstream](http://github.com/kshedden/dstream) interface.

The Godoc package documentation is [here](https://godoc.org/github.com/kshedden/statmodel/duration).

Here is an example of a proportional hazards regression:

```
// data is a Dstream containing the data
fml := "age + sex + severity + age*sex + age*severity"
dx := formula.New(fml, data).Keep("Entry", "Time", "Status").Done()
da = dstream.MemCopy(dx)

// Fit the model
ph := NewPHReg(da, "Time", "Status").Entry("Entry").Done()
rslt := ph.Fit()
fmt.Printf("%s\n", rslt.Summary())
```

Features
--------

* Cox model supports entry times (delayed entry) and stratification
