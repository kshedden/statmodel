package glm

import (
	"sort"

	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat/distuv"
)

// ScaleProfiler is used to do likelihood profile analysis on the scale
// parameter.  Set the Results field to a fitted GLMResults value.
// This is suitable for models with no additional parameters, if there
// are other parameters (e.g. in the Tweedie or Negative Binomial
// case), they are held fixed at their values from the provided fit.
type ScaleProfiler struct {

	// The profile analysis is done with respect to this fitted
	// model.
	results *GLMResults

	// After calling GetMLE, this will hold the MLE of the scale
	// parameter.
	scaleMLE float64

	// This is the largest log-likelihood value that can be
	// obtained by varying the scale parameter.
	maxLogLike float64

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
		results: result,
	}

	pa := result.Params()
	params := make([]float64, len(pa))
	copy(params, pa)
	ps.params = params

	ps.getScaleMLE()

	return ps
}

type profPoint [][2]float64

func (a profPoint) Len() int           { return len(a) }
func (a profPoint) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a profPoint) Less(i, j int) bool { return a[i][0] < a[j][1] }

// LogLike returns the profile log likelihood value at the given scale
// parameter value.
func (ps *ScaleProfiler) LogLike(scale float64) float64 {

	model := ps.results.Model().(*GLM)
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

// ScaleMLE returns the maximum likelihood estimate of the scale parameter.
func (ps *ScaleProfiler) ScaleMLE() float64 {
	return ps.scaleMLE
}

func (ps *ScaleProfiler) getScaleMLE() {

	// Center point
	scale1 := ps.results.scale
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
	ps.scaleMLE, ps.maxLogLike, hist = bisectmax(ps.LogLike, scale0, scale1, scale2, ll1)
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
	scale0 := 0.9 * ps.scaleMLE
	ll0 := ps.LogLike(scale0)
	for ll0 > ps.maxLogLike-qp {
		scale0 *= 0.9
		ll0 = ps.LogLike(scale0)
		ps.Profile = append(ps.Profile, [2]float64{scale0, ll0})
	}
	var hist [][2]float64
	scale0, hist = bisectroot(ps.LogLike, scale0, ps.scaleMLE, ll0, ps.maxLogLike, ps.maxLogLike-qp)
	ps.Profile = append(ps.Profile, hist...)

	// Right side
	scale1 := 1.1 * ps.scaleMLE
	ll1 := ps.LogLike(scale1)
	for ll1 > ps.maxLogLike-qp {
		scale1 *= 1.1
		ll1 = ps.LogLike(scale1)
		ps.Profile = append(ps.Profile, [2]float64{scale1, ll1})
	}
	scale1, hist = bisectroot(ps.LogLike, ps.scaleMLE, scale1, ps.maxLogLike, ll1, ps.maxLogLike-qp)
	ps.Profile = append(ps.Profile, hist...)

	sort.Sort(profPoint(ps.Profile))

	return scale0, scale1
}

// TweedieProfiler conducts profile likelihood analyses on a GLM with
// the Tweedie family.
type TweedieProfiler struct {

	// The profile analysis is done with respect to this fitted
	// model.
	results *GLMResults

	// The MLE of the scale parameter
	scaleMLE float64

	// The MLE of the variance power parameter
	varPowerMLE float64

	params []float64
}

// NewTweedieProfiler returns a TweedieProfiler that can be used to
// profile the variance power parameter of a Tweedie GLM.
func NewTweedieProfiler(result *GLMResults) *TweedieProfiler {

	tp := &TweedieProfiler{
		results: result,
	}

	pa := result.Params()
	tp.params = make([]float64, len(pa))
	copy(tp.params, pa)

	tp.getMLE()

	return tp
}

// ScaleMLE returns the maximum likelihood estimate of the scale parameter.
func (tp *TweedieProfiler) ScaleMLE() float64 {
	return tp.scaleMLE
}

// VarPowerMLE returns the maximum likelihood estimate of the variance power parameter..
func (tp *TweedieProfiler) VarPowerMLE() float64 {
	return tp.varPowerMLE
}

// LogLike returns the profile log likelihood value at the given
// variance power and scale parameter.
func (tp *TweedieProfiler) LogLike(pw, scale float64) float64 {

	model := tp.results.Model().(*GLM)
	model.dispersionMethod = DispersionFixed
	model.dispersionValue = scale

	model.fam = NewTweedieFamily(pw, model.link)
	copy(model.start, tp.params)
	result := model.Fit()

	return result.LogLike()
}

func (tp *TweedieProfiler) getMLE() {

	p := optimize.Problem{
		Func: func(x []float64) float64 {
			return -tp.LogLike(x[0], x[1])
		},
	}

	// Starting point for the search
	x0 := []float64{1.5, tp.results.scale}

	r, err := optimize.Minimize(p, x0, nil, &optimize.NelderMead{})
	if err != nil {
		panic(err)
	}

	tp.varPowerMLE = r.X[0]
	tp.scaleMLE = r.X[1]
}

// NegBinomProfiler conducts profile likelihood analyses on a GLM with
// the negative binomial family.
type NegBinomProfiler struct {

	// The profile analysis is done with respect to this fitted
	// model.
	results *GLMResults

	// The MLE of the dispersion parameter
	dispersionMLE float64

	// The maximum likelihood value at the MLE
	maxLogLike float64

	// A sequence of (dispersion, log-likelihood) values that lie on
	// the profile curve.
	Profile [][2]float64

	params []float64
}

// NewNegBinomProfiler returns a NegBinomProfiler that can be used to
// profile the dispersion parameter of a negative binomial GLM.
func NewNegBinomProfiler(result *GLMResults) *NegBinomProfiler {

	nb := &NegBinomProfiler{
		results: result,
	}

	pa := result.Params()
	nb.params = make([]float64, len(pa))
	copy(nb.params, pa)

	nb.getMLE()

	return nb
}

// LogLike returns the profile log likelihood value at the given
// dispersion parameter value.
func (nb *NegBinomProfiler) LogLike(disp float64) float64 {

	model := nb.results.Model().(*GLM)

	model.dispersionMethod = DispersionFixed
	model.dispersionValue = 1

	link := NewLink(LogLink)
	model.fam = NewNegBinomFamily(disp, link)
	copy(model.start, nb.params)
	result := model.Fit()

	return result.LogLike()
}

// DispersionMLE returns the maximum likelihood estimate of the dispersion parameter.
func (nb *NegBinomProfiler) DispersionMLE() float64 {
	return nb.dispersionMLE
}

func (nb *NegBinomProfiler) getMLE() {

	model := nb.results.Model().(*GLM)

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
	nb.dispersionMLE, nb.maxLogLike, hist = bisectmax(nb.LogLike, disp0, disp1, disp2, ll1)
	nb.Profile = append(nb.Profile, hist...)

	sort.Sort(profPoint(nb.Profile))
}

// ConfInt identifies dispersion parameters disp1, disp2 that define a
// profile confidence interval for the dispersion parameter.  All
// points on the profile likelihood visited during the search are
// added to the Profile field of the NegBinomProfiler value.
func (nb *NegBinomProfiler) ConfInt(prob float64) (float64, float64) {

	qp := distuv.ChiSquared{K: 1}.Quantile(prob) / 2

	// Left side
	disp0 := 0.9 * nb.dispersionMLE
	ll0 := nb.LogLike(disp0)
	for ll0 > nb.maxLogLike-qp {
		disp0 *= 0.9
		ll0 = nb.LogLike(disp0)
		nb.Profile = append(nb.Profile, [2]float64{disp0, ll0})
	}
	var hist [][2]float64
	disp0, hist = bisectroot(nb.LogLike, disp0, nb.dispersionMLE, ll0, nb.maxLogLike, nb.maxLogLike-qp)
	nb.Profile = append(nb.Profile, hist...)

	// Right side
	disp1 := 1.1 * nb.dispersionMLE
	ll1 := nb.LogLike(disp1)
	for ll1 > nb.maxLogLike-qp {
		disp1 *= 1.1
		ll1 = nb.LogLike(disp1)
		nb.Profile = append(nb.Profile, [2]float64{disp1, ll1})
	}
	disp1, hist = bisectroot(nb.LogLike, nb.dispersionMLE, disp1, nb.maxLogLike, ll1, nb.maxLogLike-qp)
	nb.Profile = append(nb.Profile, hist...)

	sort.Sort(profPoint(nb.Profile))

	return disp0, disp1
}
