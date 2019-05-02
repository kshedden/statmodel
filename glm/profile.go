package glm

import (
	"sort"

	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat/distuv"
)

// ScaleProfiler is used to do likelihood profile analysis on the scale
// parameter.  Set the Results field to a fitted GLMResults value.
// This is suitable for models with no additonal parameters, if there
// are other parameters (e.g. in the Tweedie or Negative Binomial
// case), they are held fixed at their values from the provided fit.
type ScaleProfiler struct {

	// The profile analysis is done with respect to this fitted
	// model.
	Results *GLMResults

	// After calling GetMLE, this will hold the MLE of the scale
	// parameter.
	ScaleMLE float64

	// This is the largest log-likelihood value that can be
	// obtained by varying the scale parameter.
	MaxLogLike float64

	// A sequence of (scale, log-likelihood) values that lie on
	// the profile curve.
	Profile [][2]float64

	// The parameters of the original fit.
	params []float64
}

// NewScaleProfiler returns a ScaleProfiler value that can be used to
// profile the scale parameters.
func NewScaleProfiler(result *GLMResults) *ScaleProfiler {

	ps := &ScaleProfiler{
		Results: result,
	}

	pa := result.Params()
	params := make([]float64, len(pa))
	copy(params, pa)
	ps.params = params

	return ps
}

type profPoint [][2]float64

func (a profPoint) Len() int           { return len(a) }
func (a profPoint) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a profPoint) Less(i, j int) bool { return a[i][0] < a[j][1] }

// LogLike returns the profile log likelihood value at the given scale
// parameter value.
func (ps *ScaleProfiler) LogLike(scale float64) float64 {

	model := ps.Results.Model().(*GLM)
	model.dispersionMethod = DispersionFixed

	model.dispersionValue = scale
	copy(model.start, ps.params)
	result := model.Fit()
	return result.LogLike()
}

func bisectmax(f func(float64) float64, x0, x1, x2, y1 float64) (float64, float64, [][2]float64) {

	var hist [][2]float64

	for x2-x0 > 1e-4 {
		if x2-x1 > x1-x0 {
			x := (x1 + x2) / 2
			y := f(x)
			hist = append(hist, [2]float64{x, y})
			if y > y1 {
				x0 = x1
				y1 = y
				x1 = x
			} else {
				x2 = x
			}
		} else {
			x := (x0 + x1) / 2
			y := f(x)
			hist = append(hist, [2]float64{x, y})
			if y > y1 {
				x2 = x1
				y1 = y
				x1 = x
			} else {
				x0 = x
			}
		}
	}

	return x1, y1, hist
}

func bisectroot(f func(float64) float64, x0, x1, y0, y1, yt float64) (float64, [][2]float64) {

	if (y0-yt)*(y1-yt) > 0 {
		panic("bisectroot invalid bracket")
	}

	var hist [][2]float64

	for x1-x0 > 1e-4 {
		x := (x0 + x1) / 2
		y := f(x)
		hist = append(hist, [2]float64{x, y})
		if (y-yt)*(y0-yt) > 0 {
			x0 = x
			y0 = y
		} else {
			x1 = x
		}
	}

	return (x0 + x1) / 2, hist
}

// GetScaleMLE computes the maximum likelihood estimate of the scale
// parameter and sets the ScaleMLE and MaxLogLike fields of the
// ScaleProfiler struct.
func (ps *ScaleProfiler) GetScaleMLE() {

	// Center point
	scale1 := ps.Results.scale
	ll1 := ps.LogLike(scale1)

	// Upper point
	scale2 := 1.2 * scale1
	ll2 := ps.LogLike(scale2)
	for ll2 >= ll1 {
		scale2 *= 1.2
		ll2 = ps.LogLike(scale2)
	}

	// Lower point
	scale0 := 0.8 * scale1
	ll0 := ps.LogLike(scale0)
	for ll0 >= ll1 {
		scale0 *= 0.8
		ll0 = ps.LogLike(scale0)
	}

	var hist [][2]float64
	ps.ScaleMLE, ps.MaxLogLike, hist = bisectmax(ps.LogLike, scale0, scale1, scale2, ll1)
	ps.Profile = append(ps.Profile, hist...)

	sort.Sort(profPoint(ps.Profile))
}

// ConfInt identifies scale parameters scale1, scale2 that define a
// profile confidence interval for the scale parameter.  All points on
// the profile likelihood visited during the search are added to the
// Profile field of the ScaleProfiler value.
func (ps *ScaleProfiler) ConfInt(prob float64) (float64, float64) {

	qp := distuv.ChiSquared{K: 1}.Quantile(prob) / 2

	// Left side
	scale0 := 0.9 * ps.ScaleMLE
	ll0 := ps.LogLike(scale0)
	for ll0 > ps.MaxLogLike-qp {
		scale0 *= 0.9
		ll0 = ps.LogLike(scale0)
		ps.Profile = append(ps.Profile, [2]float64{scale0, ll0})
	}
	var hist [][2]float64
	scale0, hist = bisectroot(ps.LogLike, scale0, ps.ScaleMLE, ll0, ps.MaxLogLike, ps.MaxLogLike-qp)
	ps.Profile = append(ps.Profile, hist...)

	// Right side
	scale1 := 1.1 * ps.ScaleMLE
	ll1 := ps.LogLike(scale1)
	for ll1 > ps.MaxLogLike-qp {
		scale1 *= 1.1
		ll1 = ps.LogLike(scale1)
		ps.Profile = append(ps.Profile, [2]float64{scale1, ll1})
	}
	scale1, hist = bisectroot(ps.LogLike, ps.ScaleMLE, scale1, ps.MaxLogLike, ll1, ps.MaxLogLike-qp)
	ps.Profile = append(ps.Profile, hist...)

	sort.Sort(profPoint(ps.Profile))

	return scale0, scale1
}

// TweedieProfiler conducts profile likelihood analyses on a GLM with
// the Tweedie family.
type TweedieProfiler struct {

	// The profile analysis is done with respect to this fitted
	// model.
	Results *GLMResults

	// The MLE of the scale parameter
	ScaleMLE float64

	// The MLE of the variance power parameter
	VarPowerMLE float64

	params []float64
}

// NewTweedieProfiler returns a TweedieProfiler that can be used to
// profile the variance power parameter of a Tweedie GLM.
func NewTweedieProfiler(result *GLMResults) *TweedieProfiler {

	tp := &TweedieProfiler{
		Results: result,
	}

	pa := result.Params()
	tp.params = make([]float64, len(pa))
	copy(tp.params, pa)

	return tp
}

// LogLike returns the profile log likelihood value at the given
// variance power and scale parameter.
func (tp *TweedieProfiler) LogLike(pw, scale float64) float64 {

	model := tp.Results.Model().(*GLM)
	model.dispersionMethod = DispersionFixed
	model.dispersionValue = scale

	model.fam = NewTweedieFamily(pw, model.link)
	copy(model.start, tp.params)
	result := model.Fit()

	return result.LogLike()
}

// MLE calculates the MLE of the variance power and scale parameters.
func (tp *TweedieProfiler) MLE() (float64, float64) {

	p := optimize.Problem{
		Func: func(x []float64) float64 {
			return -tp.LogLike(x[0], x[1])
		},
	}

	// Starting point for the search
	x0 := []float64{1.5, tp.Results.scale}

	r, err := optimize.Minimize(p, x0, nil, &optimize.NelderMead{})
	if err != nil {
		panic(err)
	}

	tp.VarPowerMLE = r.X[0]
	tp.ScaleMLE = r.X[1]

	return tp.VarPowerMLE, tp.ScaleMLE
}

// NegBinomProfiler conducts profile likelihood analyses on a GLM with
// the negative binomial family.
type NegBinomProfiler struct {

	// The profile analysis is done with respect to this fitted
	// model.
	Results *GLMResults

	// The MLE of the dispersion parameter
	DispersionMLE float64

	// The maximum likelihood value at the MLE
	MaxLogLike float64

	params []float64

	Profile [][2]float64
}

// NewNegBinomProfiler returns a NegBinomProfiler that can be used to
// profile the dispersion parameter of a negative binomial GLM.
func NewNegBinomProfiler(result *GLMResults) *NegBinomProfiler {

	nb := &NegBinomProfiler{
		Results: result,
	}

	pa := result.Params()
	nb.params = make([]float64, len(pa))
	copy(nb.params, pa)

	return nb
}

// LogLike returns the profile log likelihood value at the given
// dispersion parameter value.
func (nb *NegBinomProfiler) LogLike(disp float64) float64 {

	model := nb.Results.Model().(*GLM)

	model.dispersionMethod = DispersionFixed
	model.dispersionValue = 1

	link := NewLink(LogLink)
	model.fam = NewNegBinomFamily(disp, link)
	copy(model.start, nb.params)
	result := model.Fit()

	return result.LogLike()
}

// GetDispersionMLE returns the maximum likelihood estimate of the
// dispersion parameter for a negative binomial GLM.
func (nb *NegBinomProfiler) MLE() float64 {

	model := nb.Results.Model().(*GLM)

	// Center point
	disp1 := model.fam.alpha
	ll1 := nb.LogLike(disp1)

	// Upper point
	disp2 := 1.2 * disp1
	ll2 := nb.LogLike(disp2)
	for ll2 >= ll1 {
		disp2 *= 1.2
		ll2 = nb.LogLike(disp2)
	}

	// Lower point
	disp0 := 0.8 * disp1
	ll0 := nb.LogLike(disp0)
	for ll0 >= ll1 {
		disp0 *= 0.8
		ll0 = nb.LogLike(disp0)
	}

	var hist [][2]float64
	nb.DispersionMLE, nb.MaxLogLike, hist = bisectmax(nb.LogLike, disp0, disp1, disp2, ll1)
	nb.Profile = append(nb.Profile, hist...)

	sort.Sort(profPoint(nb.Profile))

	return nb.DispersionMLE
}

// ConfInt identifies dispersion parameters disp1, disp2 that define a
// profile confidence interval for the dispersion parameter.  All
// points on the profile likelihood visited during the search are
// added to the Profile field of the NegBinomProfiler value.  Always
// call MLE before calling ConfInt.
func (nb *NegBinomProfiler) ConfInt(prob float64) (float64, float64) {

	qp := distuv.ChiSquared{K: 1}.Quantile(prob) / 2

	// Left side
	disp0 := 0.9 * nb.DispersionMLE
	ll0 := nb.LogLike(disp0)
	for ll0 > nb.MaxLogLike-qp {
		disp0 *= 0.9
		ll0 = nb.LogLike(disp0)
		nb.Profile = append(nb.Profile, [2]float64{disp0, ll0})
	}
	var hist [][2]float64
	disp0, hist = bisectroot(nb.LogLike, disp0, nb.DispersionMLE, ll0, nb.MaxLogLike, nb.MaxLogLike-qp)
	nb.Profile = append(nb.Profile, hist...)

	// Right side
	disp1 := 1.1 * nb.DispersionMLE
	ll1 := nb.LogLike(disp1)
	for ll1 > nb.MaxLogLike-qp {
		disp1 *= 1.1
		ll1 = nb.LogLike(disp1)
		nb.Profile = append(nb.Profile, [2]float64{disp1, ll1})
	}
	disp1, hist = bisectroot(nb.LogLike, nb.DispersionMLE, disp1, nb.MaxLogLike, ll1, nb.MaxLogLike-qp)
	nb.Profile = append(nb.Profile, hist...)

	sort.Sort(profPoint(nb.Profile))

	return disp0, disp1
}
