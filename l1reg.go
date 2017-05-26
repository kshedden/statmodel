package statmodel

import (
	"fmt"
	"math"

	"github.com/kshedden/dstream/dstream"
)

// L1RegFitter describes a model that can be fit using L1 parameter
// regularization.
type L1RegFitter interface {

	// The log-likelihood not including L1 penalty, with scale
	// parameter set equal to 1.
	LogLike([]float64) float64

	// The score function not including L1 penalty, with scale
	// parameter set equal to 1.
	Score([]float64, []float64)

	// The Hessian matrix not including L1 penalty, with scale
	// parameter set equal to 1.
	Hessian([]float64, []float64)

	// The data to which the model is fit.
	Data() dstream.Reg

	// The L1 penalty weights.
	L1wgt() []float64

	// If true, the algorithm checks whether the local quadratic
	// approximation improves the fit, and if not, uses an
	// additional linear search.  Set to false for linear least
	// squares models.
	CheckStep() bool

	// CloneWithNewData creates a copy of the L1RegFitter,
	// replacing the DataProvider with the given value.
	CloneWithNewData(dstream.Reg) L1RegFitter
}

// FitL1Reg fits the provided L1RegFitter and returns the array of
// parameter values.
func FitL1Reg(rf L1RegFitter) []float64 {

	tol := 1e-7
	maxiter := 400

	dp := rf.Data()
	nvar := dp.NumCov()
	l1wgt := rf.L1wgt()
	params := make([]float64, nvar)
	var px float64 // l-inf parameter change

	for iter := 0; iter < maxiter; iter++ {
		px = 0
		for j := 0; j < nvar; j++ {
			focd := &dstream.FocusedReg{
				Reg:    dp,
				Col:    j,
				Params: params}

			frf := rf.CloneWithNewData(focd)
			np := opt1d(frf, params[j], l1wgt[j])
			d := math.Abs(np - params[j])
			if d > px {
				px = d
			}
			params[j] = np
		}
		if px < tol {
			break
		}
	}

	return params
}

// Use a local quadratic approximation, then fall back to a line
// search if needed.
func opt1d(rf L1RegFitter, x float64, l1wgt float64) float64 {

	v := make([]float64, 1)
	z := make([]float64, 1)
	checkstep := rf.CheckStep()

	// Quaratic approximation coefficients
	z[0] = x
	rf.Score(z, v)
	b := v[0]
	rf.Hessian(z, v)
	c := v[0]

	// The optimum point of the quadratic approximation
	d := b - c*x

	// The optimum is achieved by hard thresholding to zero
	if l1wgt > math.Abs(d) {
		return 0
	}

	// x + h is the minimizer of Q(x) + L1_wt*abs(x)
	var h float64
	if d >= 0 {
		h = (l1wgt - b) / c
	} else if d < 0 {
		h = -(l1wgt + b) / c
	} else {
		panic(fmt.Sprintf("d=%f\n", d))
	}

	// If the new point is not uphill for the target function, take it
	// and return.  This check is a bit expensive and un-necessary for
	// OLS
	if !checkstep {
		return x + h
	}
	z[0] = x
	f := rf.LogLike(z)
	z[0] = x + h
	f1 := rf.LogLike(z) + l1wgt*math.Abs(x+h)
	if f1 <= f+l1wgt*math.Abs(x)+1e-10 {
		return x + h
	}

	// Fallback for models where the loss is not quadratic
	return bisection(rf.LogLike, x-1, x+1, 1e-4)
}

// Standard bisection.
func bisection(f func([]float64) float64, xl, xu, tol float64) float64 {

	// Wrap the function so it takes a scalar argument.
	z := make([]float64, 1)
	fm := func(x float64) float64 {
		z[0] = x
		return f(z)
	}
	var x0, x1, x2, f0, f1, f2 float64
	x0 = xl
	x2 = xu

	q := false
	for k := 0; k < 10; k++ {
		f0 = fm(x0)
		f2 = fm(x2)
		x1 = (x0 + x2) / 2
		f1 = fm(x1)

		if f1 < f0 && f1 < f2 {
			q = true
			break
		}
		x0 -= 1
		x2 += 1
	}

	if !q {
		msg := fmt.Sprintf("Failed to find bracket:\nx0=%f x1=%f x2=%f f0=%f f1=%f f2=%f\n", x0, x1, x2, f0, f1, f2)
		panic(msg)
	}

	for x2-x0 > tol {
		if x1-x0 > x2-x1 {
			xx := (x0 + x1) / 2
			ff := fm(xx)
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
			ff := fm(xx)
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
