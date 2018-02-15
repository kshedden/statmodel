package statmodel

import (
	"fmt"
	"math"

	"github.com/kshedden/dstream/dstream"
)

// A FocusCreator derives a focusable model from a general regression model.
type FocusCreator interface {

	// Returns a focusable copy of the parent model
	GetFocusable() ModelFocuser

	// Returns the number of coefficients in the original
	// (unfocused) model.
	NumParams() int
}

// A ModelFocuser is a regression model that can be focused on one
// covariate, placing the effects of all other covariates into the
// offset.
type ModelFocuser interface {

	// Log-likelihood of the full model
	LogLike(Parameter) float64

	// Score function of the full model
	Score(Parameter, []float64)

	// Hessian matrix of the full model
	Hessian(Parameter, HessType, []float64)

	// The number of covariates in the model
	NumParams() int

	// Focus the model on the given covariate, using the given
	// coefficients.
	Focus(int, []float64)

	// The dataset used to fit the model
	DataSet() dstream.Dstream
}

// model1d is a convenience class that modifies the signatures of the
// likelihood, score and Hessian functions for single parameter
// models.
type model1d struct {
	model ModelFocuser
	param Parameter
}

func (m1 *model1d) LogLike(x float64) float64 {
	m1.param.SetCoeff([]float64{x})
	return m1.model.LogLike(m1.param)
}

func (m1 *model1d) Score(x float64) float64 {
	m1.param.SetCoeff([]float64{x})
	s := []float64{0}
	m1.model.Score(m1.param, s)
	return s[0]
}

func (m1 *model1d) Hessian(x float64) float64 {
	m1.param.SetCoeff([]float64{x})
	h := []float64{0}
	m1.model.Hessian(m1.param, ObsHess, h)
	return h[0]
}

// FitL1Reg fits the provided L1RegFitter and returns the array of
// parameter values.
func FitL1Reg(rf FocusCreator, param Parameter, l1wgt []float64, xn []float64, checkstep bool) Parameter {

	tol := 1e-7
	maxiter := 400

	// A parameter for the 1-d focused model, sharing all
	// non-coefficient parameters.
	param1d := param.Clone()
	param1d.SetCoeff([]float64{0})

	rf1 := rf.GetFocusable()
	nvar := rf.NumParams()
	m1 := model1d{rf1, param1d}

	coeff := param.GetCoeff()

	// Outer coordinate descent loop.
	for iter := 0; iter < maxiter; iter++ {

		// L-inf of the increment in the parameter vector
		px := 0.0

		// Loop over covariates
		for j := 0; j < nvar; j++ {

			// Get the new point
			rf1.Focus(j, coeff)
			np := opt1d(m1, coeff[j], l1wgt[j], checkstep)

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

	for j := range coeff {
		coeff[j] /= xn[j]
	}

	return param
}

// Use a local quadratic approximation, then fall back to a line
// search if needed.
func opt1d(m1 model1d, coeff float64, l1wgt float64, checkstep bool) float64 {

	// Quadratic approximation coefficients
	b := -m1.Score(coeff)
	c := -m1.Hessian(coeff)

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
	f0 := -m1.LogLike(coeff) + l1wgt*math.Abs(coeff)
	f1 := -m1.LogLike(coeff+h) + l1wgt*math.Abs(coeff+h)
	if f1 <= f0+1e-10 {
		return coeff + h
	}

	// Wrap the log-likelihood so it takes a scalar argument.
	fw := func(z float64) float64 {
		f := -m1.LogLike(z) + l1wgt*math.Abs(z)
		return f
	}

	// Fallback for models where the loss is not quadratic
	np := bisection(fw, coeff-1, coeff+1, 1e-4)
	return np
}

// Standard bisection to minimize f.
func bisection(f func(float64) float64, xl, xu, tol float64) float64 {

	var x0, x1, x2, f0, f1, f2 float64

	// Try to find a bracket.
	success := false
	x0, x2 = xl, xu
	for k := 0; k < 100; k++ {
		f0 = f(x0)
		f2 = f(x2)
		x1 = (x0 + x2) / 2
		f1 = f(x1)

		if f1 < f0 && f1 < f2 {
			success = true
			break
		}
		x0 -= 1
		x2 += 1
	}

	if !success {
		msg := fmt.Sprintf("Failed to find bracket:\nx0=%f x1=%f x2=%f f0=%f f1=%f f2=%f\n", x0, x1, x2, f0, f1, f2)
		panic(msg)
	}

	for x2-x0 > tol {
		if x1-x0 > x2-x1 {
			xx := (x0 + x1) / 2
			ff := f(xx)
			if ff < f1 {
				x2 = x1
				f2 = f1
				x1 = xx
				f1 = ff
			} else {
				x0 = xx
				f0 = ff
			}
		} else {
			xx := (x1 + x2) / 2
			ff := f(xx)
			if ff < f1 {
				x0 = x1
				f0 = f1
				x1 = xx
				f1 = ff
			} else {
				x2 = xx
				f2 = ff
			}
		}
	}

	return x1
}
