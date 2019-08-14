package statmodel

import (
	"fmt"
	"math"
)

// Focuser restricts a model to one parameter.
type Focuser interface {
	NumParams() int
	NumObs() int
	Focus(int, []float64, []float64) RegFitter
	LogLike(Parameter, bool) float64
	Score(Parameter, []float64)
	Hessian(Parameter, HessType, []float64)
}

// FitL1Reg fits the provided L1RegFitter and returns the array of
// parameter values.
func FitL1Reg(model Focuser, param Parameter, l1wgt, offset []float64, checkstep bool) Parameter {

	maxiter := 400

	// A parameter for the 1-d focused model.
	param1d := param.Clone()
	param1d.SetCoeff([]float64{0})

	nvar := model.NumParams()
	nobs := model.NumObs()

	// Since we are using non-normalized log-likelihood, the
	// tolerance can scale with the sample size.
	tol := 1e-7 * float64(nobs)
	if tol > 0.1 {
		tol = 0.1
	}

	coeff := param.GetCoeff()

	// Outer coordinate descent loop.
	for iter := 0; iter < maxiter; iter++ {

		// L-inf of the increment in the parameter vector
		px := 0.0

		// Loop over covariates
		for j := 0; j < nvar; j++ {

			// Get the new point
			fmodel := model.Focus(j, coeff, offset)
			np := opt1d(fmodel, coeff[j], param1d, float64(nobs)*l1wgt[j], checkstep)

			// Update the change measure
			d := math.Abs(np - coeff[j])
			if d > px {
				px = d
			}

			coeff[j] = np
		}

		if px < tol {
			break
		}
	}

	return param
}

// Use a local quadratic approximation, then fall back to a line
// search if needed.
func opt1d(m1 RegFitter, coeff float64, par Parameter, l1wgt float64, checkstep bool) float64 {

	// Quadratic approximation coefficients
	bv := make([]float64, 1)
	par.SetCoeff([]float64{coeff})
	m1.Score(par, bv)
	b := -bv[0]
	cv := make([]float64, 1)
	m1.Hessian(par, ObsHess, cv)
	c := -cv[0]

	// The optimum point of the quadratic approximation
	d := b - c*coeff

	if l1wgt > math.Abs(d) {
		// The optimum is achieved by hard thresholding to zero
		return 0
	}

	// pj + h is the minimizer of Q(x) + L1_wt*abs(x)
	var h float64
	if d >= 0 {
		h = (l1wgt - b) / c
	} else if d < 0 {
		h = -(l1wgt + b) / c
	} else {
		panic(fmt.Sprintf("d=%f\n", d))
	}

	if !checkstep {
		return coeff + h
	}

	// Check whether the new point improves the target function.
	// This check is a bit expensive and not necessary for OLS
	par.SetCoeff([]float64{coeff})
	f0 := -m1.LogLike(par, false) + l1wgt*math.Abs(coeff)
	par.SetCoeff([]float64{coeff + h})
	f1 := -m1.LogLike(par, false) + l1wgt*math.Abs(coeff+h)
	if f1 <= f0+1e-10 {
		return coeff + h
	}

	// Wrap the log-likelihood so it takes a scalar argument.
	fw := func(z float64) float64 {
		par.SetCoeff([]float64{z})
		f := -m1.LogLike(par, false) + l1wgt*math.Abs(z)
		return f
	}

	// Fallback for models where the loss is not quadratic
	w := 1.0
	btol := 1e-7
	np := bisection(fw, coeff-w, coeff+w, btol)
	return np
}

// Standard bisection to minimize f.
func bisection(f func(float64) float64, xl, xu, tol float64) float64 {

	var x0, x1, x2, f0, f1, f2 float64

	// Try to find a bracket.
	success := false
	x0, x2 = xl, xu
	x1 = (x0 + x2) / 2
	f1 = f(x1)
	for k := 0; k < 100; k++ {

		// TODO recomputing some values here
		f0 = f(x0)
		f1 = f(x1)
		f2 = f(x2)

		if f1 < f0 && f1 < f2 {
			success = true
			break
		}

		if f0 > f1 && f1 > f2 {
			// Slide right
			x0 = x1
			x1 = x2
			x2 += 1.5 * (x1 - x0)
			continue
		}

		if f0 < f1 && f1 < f2 {
			// Slide left
			x1 = x0
			x2 = x1
			x0 -= 1.5 * (x2 - x1)
			continue
		}

		x0 = x1 - 2*(x1-x0)
		x2 = x1 + 2*(x2-x1)
	}

	if !success {
		fmt.Printf("Did not find bracket...\n")
		if f0 < f1 && f0 < f2 {
			return x0
		} else if f1 < f0 && f1 < f2 {
			return x1
		} else {
			return x2
		}
	}

	iter := 0
	for x2-x0 > tol {
		iter++
		if x1-x0 > x2-x1 {
			xx := (x0 + x1) / 2
			ff := f(xx)
			if ff < f1 {
				x2 = x1
				x1, f1 = xx, ff
			} else {
				x0 = xx
			}
		} else {
			xx := (x1 + x2) / 2
			ff := f(xx)
			if ff < f1 {
				x0 = x1
				x1, f1 = xx, ff
			} else {
				x2 = xx
			}
		}
	}

	return x1
}
