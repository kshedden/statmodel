The __duration__ package supports fitting statistical models for event
time data (also known as "survival analysis") in Go.  Currently, Cox
proportional hazards regression, Kaplan-Meier estimates of the
marginal survival function, and estimation of the cumulative incidence
function are supported.  The analysis functions accept data using the
[dstream](http://github.com/kshedden/dstream) interface.

Link to the [godoc documentation](https://godoc.org/github.com/kshedden/duration).

Here is an example of a proportional hazards regression:

```
// Provide an io.Reader to the CSV-formatted data source.
da := dstream.FromCSV(r).AllFloat().HasHeader().Done()

// Prepare a design matrix using a formula
fml := "age + sex + severity + age*sex + age*severity"
dx := formula.New(fml, dx).Keep([]string{"Entry", "Time", "Status"}).Done()
da = dstream.MemCopy(dx)

// Fit the model
ph := NewPHReg(da, "Time", "Status").Entry("Entry").Done()
rslt := ph.Fit()
fmt.Printf("%s\n", rslt.Summary())
```

Features
--------

* Cox model supports entry times (delayed entry) and stratification

* Cox model uses the Breslow log-likelihood