package glm

import (
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"sync"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/optimize"

	"github.com/kshedden/statmodel/statmodel"
)

// GLM represents a generalized linear model.
type GLM struct {

	// Names of the variables (including response, covariates, weights, offset, etc.)
	// The order agrees with the order of data.
	varnames []string

	// The data, stored by column.
	data [][]statmodel.Dtype

	// Positions of the covariates in data.
	xpos []int

	// Position of the outcome variable
	ypos int

	// Position of the offset variable, -1 if not present.
	offsetpos int

	// Position of the weight variable, -1 if not present.
	weightpos int

	// The GLM family
	fam *Family

	// The GLM link function
	link *Link

	// The GLM variance function
	vari *Variance

	// Either IRLS (default if L1 weights are not present),
	// coordinate (if L1 weights are present, or gradient.
	fitMethod string

	// Starting values, optional
	start []float64

	// L1 (Lasso) penalty weights, optional.  FitMethod is
	// ignored if present.
	l1wgtMap map[string]float64
	l1wgt    []float64

	// L2 (ridge) penalty weights, optional.  FitMethod is
	// ignored if present.
	l2wgtMap map[string]float64
	l2wgt    []float64

	// Optimization settings
	settings *optimize.Settings

	// Optimization method
	method optimize.Method

	// If not nil, write log messages here
	log *log.Logger

	// Use concurrent calculations in IRLS if the chunk size is at least
	// as large as this value.
	concurrentIRLS int

	// The approach to handling the dispersion parameter
	dispersionMethod DispersionForm

	// If the dispersion is fixed, it is held at this value.
	dispersionValue float64

	// A pool of n-dimensional slices
	nslices [][]float64
}

func (model *GLM) putNslice(x []float64) {
	model.nslices = append(model.nslices, x)
}

func (model *GLM) getNslice() []float64 {

	if len(model.nslices) == 0 {
		return make([]float64, model.NumObs())
	}
	q := len(model.nslices) - 1
	x := model.nslices[q]
	zero(x)
	model.nslices = model.nslices[0:q]

	return x
}

// DispersionForm indicates an approach for handling the dispersion parameter.
type DispersionForm uint8

// DispersionFixed, ... define ways to handle the dispersion parameter in a GLM.
const (
	dispersionUnknown DispersionForm = iota
	DispersionFixed
	DispersionFree
	DispersionEstimate
)

// GLMParams represents the model parameters for a GLM.
type GLMParams struct {
	coeff []float64
	scale float64
}

// GetCoeff returns the coefficients (slopes for individual
// covariates) from the parameter.
func (p *GLMParams) GetCoeff() []float64 {
	return p.coeff
}

// SetCoeff sets the coefficients (slopes for individual covariates)
// for the parameter.
func (p *GLMParams) SetCoeff(coeff []float64) {
	p.coeff = coeff
}

// Clone produces a deep copy of the parameter value.
func (p *GLMParams) Clone() statmodel.Parameter {
	coeff := make([]float64, len(p.coeff))
	copy(coeff, p.coeff)
	return &GLMParams{
		coeff: coeff,
		scale: p.scale,
	}
}

// NumParams returns the number of covariates in the model.
func (model *GLM) NumParams() int {
	return len(model.xpos)
}

// Xpos returns the positions of the covariates in the model's data
// stream.
func (model *GLM) Xpos() []int {
	return model.xpos
}

// Dataset returns the data columns that are used to fit the model.
func (model *GLM) Dataset() [][]statmodel.Dtype {
	return model.data
}

// ConcurrentIRLS sets the minimum chunk size for which concurrent
// calculations are used during IRLS.
func (model *GLM) ConcurrentIRLS(n int) *GLM {
	model.concurrentIRLS = n
	return model
}

// GLMResults describes the results of a fitted generalized linear model.
type GLMResults struct {
	statmodel.BaseResults

	scale float64
}

// Scale returns the estimated scale parameter.
func (rslt *GLMResults) Scale() float64 {
	return rslt.scale
}

// Config defines configuration parameters for a GLM.
type Config struct {

	// A logger to which logging information is wreitten
	Log *log.Logger

	// FitMethod is the numerical approach for fitting the model.  Allowed
	// values include IRLS, gradient, and coordinate.
	FitMethod string

	// ConcurrentIRLS is the number of concurrent goroutines used in IRLS
	// fitting.
	ConcurrentIRLS int

	// Start contains starting values for the regression parameter estimates
	Start []float64

	// WeightVar is the name of the variable for frequency-weighting the cases, if an empty
	// string, all weights are equal to 1.
	WeightVar string

	// OffsetVar is the name of a variable providing an offset
	OffsetVar string

	// Family defines a GLMfamily.
	Family *Family

	// Link defines a GLM link function; if not provided the default link for the family is used.
	Link *Link

	// VarFunc defines how the variance relates to the mean; if not provided, the default
	// variance function for the family is used.
	VarFunc *Variance

	// L1Penalty gives the level of penalization for each variable, by name
	L1Penalty map[string]float64

	// L2Penalty gives the level of penalization for each variable, by name
	L2Penalty map[string]float64

	// DispersionForm determines how the dispersion parameter is handled
	DispersionForm DispersionForm
}

// DefaultConfig returns default configuration values for a GLM.
func DefaultConfig() *Config {

	return &Config{
		Family:         NewFamily(GaussianFamily),
		FitMethod:      "IRLS",
		ConcurrentIRLS: 1000,
	}
}

func checkValid(data statmodel.Dataset) error {

	for k, na := range data.Names() {
		if len(strings.TrimSpace(na)) == 0 {
			msg := fmt.Sprintf("The variable in position %d has an invalid name.", k)
			return fmt.Errorf(msg)
		}
	}

	return nil
}

// NewGLM creates a new generalized linear model for the given family,
// using its default link and variance functions.
func NewGLM(data statmodel.Dataset, outcome string, predictors []string, config *Config) (*GLM, error) {

	if config == nil {
		config = DefaultConfig()
	}

	if err := checkValid(data); err != nil {
		return nil, err
	}

	pos := make(map[string]int)
	for i, v := range data.Names() {
		pos[v] = i
	}

	ypos, ok := pos[outcome]
	if !ok {
		msg := fmt.Sprintf("Outcome variable '%s' not found in dataset\n", outcome)
		return nil, fmt.Errorf(msg)
	}

	var xpos []int
	for _, xna := range predictors {
		xp, ok := pos[xna]
		if !ok {
			msg := fmt.Sprintf("Predictor '%s' not found in dataset\n", xna)
			return nil, fmt.Errorf(msg)
		}
		xpos = append(xpos, xp)
	}

	weightpos := -1
	if config.WeightVar != "" {
		var ok bool
		weightpos, ok = pos[config.WeightVar]
		if !ok {
			msg := fmt.Sprintf("Weight variable '%s' not found in dataset\n", config.WeightVar)
			return nil, fmt.Errorf(msg)
		}
	}

	offsetpos := -1
	if config.OffsetVar != "" {
		var ok bool
		offsetpos, ok = pos[config.OffsetVar]
		if !ok {
			msg := fmt.Sprintf("Offset variable '%s' not found in dataset\n", config.OffsetVar)
			return nil, fmt.Errorf(msg)
		}
	}

	varnames := data.Names()

	penToSlice := func(m map[string]float64) []float64 {
		if m == nil || len(m) == 0 {
			return nil
		}
		v := make([]float64, len(xpos))
		for j, k := range xpos {
			v[j] = m[varnames[k]]
		}
		return v
	}

	model := &GLM{
		data:             data.Data(),
		varnames:         data.Names(),
		ypos:             ypos,
		xpos:             xpos,
		weightpos:        weightpos,
		offsetpos:        offsetpos,
		dispersionMethod: config.DispersionForm,
		fitMethod:        config.FitMethod,
		concurrentIRLS:   config.ConcurrentIRLS,
		fam:              config.Family,
		link:             config.Link,
		vari:             config.VarFunc,
		start:            config.Start,
		l1wgt:            penToSlice(config.L1Penalty),
		l2wgt:            penToSlice(config.L2Penalty),
		l1wgtMap:         config.L1Penalty,
		l2wgtMap:         config.L2Penalty,
		log:              config.Log,
	}

	model.init()

	return model, nil
}

func (model *GLM) setup() {

	if model.link == nil {
		li := model.fam.validLinks[0]
		if model.log != nil {
			model.log.Printf("Using default link for family: %v\n", li)
		}
		model.link = NewLink(li)
	}

	if model.vari == nil {
		// Set a default variance function
		switch model.fam.TypeCode {
		case BinomialFamily:
			model.vari = NewVariance(BinomialVar)
		case PoissonFamily:
			model.vari = NewVariance(IdentityVar)
		case QuasiPoissonFamily:
			model.vari = NewVariance(IdentityVar)
		case GaussianFamily:
			model.vari = NewVariance(ConstantVar)
		case GammaFamily:
			model.vari = NewVariance(SquaredVar)
		case InvGaussianFamily:
			model.vari = NewVariance(CubedVar)
		case NegBinomFamily:
			model.vari = NewNegBinomVariance(model.fam.alpha)
		case TweedieFamily:
			model.vari = NewTweedieVariance(model.fam.alpha)
		default:
			msg := fmt.Sprintf("Unknown GLM family: %s\n", model.fam.Name)
			panic(msg)
		}
		if model.log != nil {
			model.log.Printf("Using default variance for family: %v\n", model.vari)
		}
	}
}

func (model *GLM) check() {

	if model.l1wgt != nil && len(model.l1wgt) != len(model.xpos) {
		msg := fmt.Sprintf("GLM: The L1 weight vector has length %d, but the model has %d covariates.\n",
			len(model.l1wgt), len(model.xpos))
		panic(msg)
	}

	if model.l2wgt != nil && len(model.l2wgt) != len(model.xpos) {
		msg := fmt.Sprintf("GLM: The L2 weight vector has length %d, but the model has %d covariates.\n",
			len(model.l2wgt), len(model.xpos))
		panic(msg)
	}
}

func (model *GLM) init() *GLM {

	if model.fam == nil {
		msg := "A GLM family must be specified.\n"
		panic(msg)
	}

	model.setupDispersion()
	model.setupPenalty()
	model.setup()

	if len(model.start) == 0 {
		model.start = make([]float64, model.NumParams())
	}

	model.check()

	return model
}

func (model *GLM) setupDispersion() {
	if model.dispersionMethod == dispersionUnknown {
		model.dispersionMethod = model.fam.dispersionDefaultMethod
		if model.dispersionMethod == DispersionFixed {
			model.dispersionValue = model.fam.dispersionDefaultValue
		}
	}

	if model.dispersionMethod == DispersionFixed && model.dispersionValue <= 0 {
		panic("A fixed dispersion value must be a positive number.")
	}
}

func (model *GLM) setupPenalty() {

	f := func(mp map[string]float64) []float64 {
		wvec := make([]float64, len(model.xpos))
		for i, j := range model.xpos {
			wvec[i] = mp[model.varnames[j]]
		}
		return wvec
	}

	if model.l1wgtMap != nil {
		model.l1wgt = f(model.l1wgtMap)
	}

	if model.l2wgtMap != nil {
		model.l2wgt = f(model.l2wgtMap)
	}
}

// SetFamily is a convenience method that sets the family, link, and
// variance function based on the given family name.  The link and
// variance functions are set to their canonical values.
func (model *GLM) SetFamily(fam FamilyType) *GLM {

	switch fam {
	case BinomialFamily:
		model.fam = &binomial
		model.link = NewLink(LogitLink)
		model.vari = NewVariance(BinomialVar)
	case PoissonFamily:
		model.fam = &poisson
		model.link = NewLink(LogLink)
		model.vari = NewVariance(IdentityVar)
	case QuasiPoissonFamily:
		model.fam = &quasiPoisson
		model.link = NewLink(LogLink)
		model.vari = NewVariance(IdentityVar)
	case GaussianFamily:
		model.fam = &gaussian
		model.link = NewLink(IdentityLink)
		model.vari = NewVariance(ConstantVar)
	case GammaFamily:
		model.fam = &gamma
		model.link = NewLink(RecipLink)
		model.vari = NewVariance(SquaredVar)
	case InvGaussianFamily:
		model.fam = &invGaussian
		model.link = NewLink(RecipSquaredLink)
		model.vari = NewVariance(CubedVar)
	case NegBinomFamily:
		panic("GLM: can't set family to NegBinom using SetFamily")
	case TweedieFamily:
		// TODO something here?
	default:
		msg := fmt.Sprintf("Unknown GLM family: %v\n", fam)
		panic(msg)
	}

	return model
}

// LogLike returns the log-likelihood value for the generalized linear
// model at the given parameter values.  If exact is false, multiplicative
// factors that are constant with respect to the parameter may be omitted.
func (model *GLM) LogLike(params statmodel.Parameter, exact bool) float64 {

	gpar := params.(*GLMParams)
	coeff := gpar.coeff
	scale := gpar.scale

	nobs := model.NumObs()
	linpred := model.getNslice()
	mn := model.getNslice()
	var wgts, off []statmodel.Dtype

	yda := model.data[model.ypos]

	if model.weightpos != -1 {
		wgts = model.data[model.weightpos]
	}
	if model.offsetpos != -1 {
		off = model.data[model.offsetpos]
	}

	// Update the linear predictor
	for j, k := range model.xpos {
		xda := model.data[k]
		for i := range linpred {
			linpred[i] += float64(xda[i]) * coeff[j]
		}
	}
	if off != nil {
		for i := range linpred {
			linpred[i] += float64(off[i])
		}
	}

	// Update the log likelihood value
	model.link.InvLink(linpred, mn)
	loglike := model.fam.LogLike(yda, mn, wgts, scale, exact)

	// Account for the L2 penalty
	if model.l2wgt != nil {
		for j, v := range model.l2wgt {
			loglike -= float64(nobs) * v * coeff[j] * coeff[j] / 2
		}
	}

	model.putNslice(linpred)
	model.putNslice(mn)

	return loglike
}

func scoreFactor(yda []statmodel.Dtype, mn, deriv, va, sfac []float64) {
	for i, y := range yda {
		sfac[i] = (float64(y) - mn[i]) / (deriv[i] * va[i])
	}
}

// Score evaluates the score function for the GLM at the given
// parameter values.
func (model *GLM) Score(params statmodel.Parameter, score []float64) {

	gpar := params.(*GLMParams)
	coeff := gpar.coeff
	scale := gpar.scale

	var wgts, off []statmodel.Dtype

	zero(score)

	yda := model.data[model.ypos]
	linpred := model.getNslice()
	mn := model.getNslice()
	deriv := model.getNslice()
	va := model.getNslice()
	fac := model.getNslice()

	if model.weightpos != -1 {
		wgts = model.data[model.weightpos]
	}
	if model.offsetpos != -1 {
		off = model.data[model.offsetpos]
	}

	// Update the linear predictor
	for j, k := range model.xpos {
		xda := model.data[k]
		for i := range linpred {
			linpred[i] += float64(xda[i]) * coeff[j]
		}
	}
	if off != nil {
		for i := range linpred {
			linpred[i] += float64(off[i])
		}
	}

	model.link.InvLink(linpred, mn)
	model.link.Deriv(mn, deriv)
	model.vari.Var(mn, va)

	scoreFactor(yda, mn, deriv, va, fac)

	for j, k := range model.xpos {

		xda := model.data[k]

		if wgts == nil {
			for i := range xda {
				score[j] += fac[i] * float64(xda[i])
			}
		} else {
			for i := range xda {
				score[j] += fac[i] * float64(wgts[i]) * float64(xda[i])
			}
		}
	}

	if scale != 1 {
		floats.Scale(1/scale, score)
	}

	// Account for the L2 penalty
	if model.l2wgt != nil {
		nobs := float64(len(linpred))
		for j, v := range model.l2wgt {
			score[j] -= nobs * v * coeff[j]
		}
	}

	model.putNslice(linpred)
	model.putNslice(mn)
	model.putNslice(deriv)
	model.putNslice(va)
	model.putNslice(fac)
}

// Hessian returns the Hessian matrix for the model.  The Hessian is
// returned as a one-dimensional array, which is the vectorized form
// of the Hessian matrix.  Either the observed or expected Hessian can
// be calculated.
func (model *GLM) Hessian(param statmodel.Parameter, ht statmodel.HessType, hess []float64) {

	gpar := param.(*GLMParams)
	coeff := gpar.coeff

	yda := model.data[model.ypos]
	nobs := len(yda)
	nvar := model.NumParams()
	xdat := make([][]statmodel.Dtype, nvar)
	linpred := model.getNslice()
	mn := model.getNslice()
	lderiv := model.getNslice()
	lderiv2 := model.getNslice()
	va := model.getNslice()
	fac := model.getNslice()
	vad := model.getNslice()
	sfac := model.getNslice()
	zero(hess)

	for j, k := range model.xpos {
		xdat[j] = model.data[k]
	}

	var wgts, off []statmodel.Dtype

	if model.weightpos != -1 {
		wgts = model.data[model.weightpos]
	}
	if model.offsetpos != -1 {
		off = model.data[model.offsetpos]
	}

	// Update the linear predictor
	zero(linpred)
	for j := range model.xpos {
		for i := range linpred {
			linpred[i] += coeff[j] * float64(xdat[j][i])
		}
	}
	if off != nil {
		for i := range linpred {
			linpred[i] += float64(off[i])
		}
	}

	// The mean response
	model.link.InvLink(linpred, mn)

	model.link.Deriv(mn, lderiv)
	model.vari.Var(mn, va)

	// Factor for the expected Hessian
	for i := 0; i < len(lderiv); i++ {
		fac[i] = 1 / (lderiv[i] * lderiv[i] * va[i])
	}

	// Adjust the factor for the observed Hessian
	if ht == statmodel.ObsHess {
		model.link.Deriv2(mn, lderiv2)
		model.vari.Deriv(mn, vad)
		scoreFactor(yda, mn, lderiv, va, sfac)

		for i := range fac {
			h := va[i]*lderiv2[i] + lderiv[i]*vad[i]
			h *= sfac[i] * fac[i]
			if wgts != nil {
				h *= float64(wgts[i])
			}
			fac[i] *= 1 + h
		}
	}

	// Update the Hessian matrix
	model.hessXprod(xdat, fac, wgts, hess)

	// Fill in the upper triangle
	for j1 := range model.xpos {
		for j2 := 0; j2 < j1; j2++ {
			hess[j2*nvar+j1] = hess[j1*nvar+j2]
		}
	}

	// Account for the L2 penalty
	if model.l2wgt != nil {
		for j, v := range model.l2wgt {
			hess[j*nvar+j] -= float64(nobs) * v
		}
	}

	model.putNslice(linpred)
	model.putNslice(mn)
	model.putNslice(lderiv)
	model.putNslice(lderiv2)
	model.putNslice(va)
	model.putNslice(fac)
	model.putNslice(vad)
	model.putNslice(sfac)
}

func (model *GLM) hessXprod(xdat [][]statmodel.Dtype, fac []float64, wgts []statmodel.Dtype, hess []float64) {

	nvar := len(xdat)

	var wg sync.WaitGroup

	for j1 := range model.xpos {
		for j2 := 0; j2 <= j1; j2++ {

			wg.Add(1)
			go func(j1, j2 int) {
				x1 := xdat[j1]
				x2 := xdat[j2]
				if wgts == nil {
					for i := range x1 {
						hess[j1*nvar+j2] -= fac[i] * float64(x1[i]*x2[i])
					}
				} else {
					for i := range x1 {
						hess[j1*nvar+j2] -= fac[i] * float64(x1[i]*x2[i]*wgts[i])
					}
				}
				wg.Done()
			}(j1, j2)
		}
	}

	wg.Wait()
}

// Focus returns a new GLM instance with a single variable, which is variable j in the
// original model.  The effects of the remaining covariates are captured
// through the offset.
func (model *GLM) Focus(pos int, coeff []float64, offset []float64) statmodel.RegFitter {

	fmodel := *model

	fmodel.varnames = []string{model.varnames[model.ypos], model.varnames[model.xpos[pos]]}
	fmodel.data = [][]statmodel.Dtype{model.data[model.ypos], model.data[model.xpos[pos]]}
	fmodel.xpos = []int{1}
	fmodel.ypos = 0
	fmodel.start = nil
	fmodel.settings = nil
	fmodel.settings = nil
	fmodel.method = nil
	fmodel.log = model.log
	fmodel.concurrentIRLS = 0

	if model.weightpos != -1 {
		fmodel.varnames = append(fmodel.varnames, model.varnames[model.weightpos])
		fmodel.data = append(fmodel.data, model.data[model.weightpos])
		fmodel.weightpos = len(fmodel.data) - 1
	}

	// Allocate a new slice for the offset
	nobs := model.NumObs()
	if cap(offset) < nobs {
		offset = make([]float64, nobs)
	} else {
		offset = offset[0:nobs]
		zero(offset)
	}
	fmodel.varnames = append(fmodel.varnames, "__offset")
	fmodel.data = append(fmodel.data, make([]statmodel.Dtype, model.NumObs()))
	fmodel.offsetpos = len(fmodel.data) - 1

	// Fill in the offset
	off := fmodel.data[fmodel.offsetpos]
	zerodtype(off)
	for j, k := range model.xpos {
		if j != pos {
			for i := range off {
				off[i] += statmodel.Dtype(coeff[j] * float64(model.data[k][i]))
			}
		}
	}

	if model.offsetpos != -1 {
		offsetOrig := model.data[model.offsetpos]
		for i := range offsetOrig {
			off[i] += offsetOrig[i]
		}
	}

	if model.l2wgtMap != nil {
		fmodel.l2wgtMap = make(map[string]float64)
		vn := model.varnames[model.xpos[pos]]
		fmodel.l2wgtMap[vn] = model.l2wgtMap[vn]
		fmodel.l2wgt = []float64{model.l2wgtMap[vn]}
	} else {
		fmodel.l2wgt = nil
	}

	fmodel.l1wgtMap = nil
	fmodel.l1wgt = nil

	return &fmodel
}

// NumObs returns the number of observations used to fit the model.
func (model *GLM) NumObs() int {
	return len(model.data[0])
}

// fitRegularized estimates the parameters of the GLM using L1
// regularization (with optimal L2 regularization).  This invokes
// coordinate descent optimization.  For fitting with no L1
// regularization (with or without L2 regularization), call
// fitGradient which invokes gradient optimization.
func (model *GLM) fitRegularized() *GLMResults {

	if model.log != nil {
		model.log.Print("Regularized fitting\n")
	}

	start := &GLMParams{
		coeff: model.start,
		scale: 1.0,
	}

	checkstep := strings.ToLower(model.fam.Name) != "gaussian"
	offset := make([]float64, model.NumObs())
	par := statmodel.FitL1Reg(model, start, model.l1wgt, offset, checkstep)
	coeff := par.GetCoeff()

	// Covariate names
	var xna []string
	for _, j := range model.xpos {
		xna = append(xna, model.varnames[j])
	}

	scale := model.EstimateScale(coeff)

	results := &GLMResults{
		BaseResults: statmodel.NewBaseResults(model, 0, coeff, xna, nil),
		scale:       scale,
	}

	return results
}

// Fit estimates the parameters of the GLM and returns a results
// object.  Unregularized fits and fits involving L2 regularization
// can be obtained, but if L1 regularization is desired use
// FitRegularized instead of Fit.
func (model *GLM) Fit() *GLMResults {

	if model.l1wgt != nil {
		return model.fitRegularized()
	}

	nvar := model.NumParams()
	maxiter := 20

	var start []float64
	if model.start != nil {
		start = model.start
	} else {
		start = make([]float64, nvar)
	}

	if model.l2wgt != nil {
		model.fitMethod = "gradient"
	}

	var params []float64

	if strings.ToLower(model.fitMethod) == "gradient" {
		if model.log != nil {
			model.log.Print("Unregularized fitting using gradient optimization\n")
		}
		params, _ = model.fitGradient(start)
	} else {
		if model.log != nil {
			model.log.Print("Unregularized fitting using IRLS\n")
		}
		params = model.fitIRLS(start, maxiter)
	}

	scale := model.EstimateScale(params)

	vcov, _ := statmodel.GetVcov(model, &GLMParams{params, scale})
	floats.Scale(scale, vcov)

	ll := model.LogLike(&GLMParams{params, scale}, true)

	var xna []string
	for _, j := range model.xpos {
		xna = append(xna, model.varnames[j])
	}

	results := &GLMResults{
		BaseResults: statmodel.NewBaseResults(model, ll, params, xna, vcov),
		scale:       scale,
	}

	return results
}

// fitGradient uses gradient-based optimization to obtain the fitted
// GLM parameters.
func (model *GLM) fitGradient(start []float64) ([]float64, float64) {

	p := optimize.Problem{
		Func: func(x []float64) float64 {
			return -model.LogLike(&GLMParams{x, 1}, false)
		},
		Grad: func(grad, x []float64) {
			if len(grad) != len(x) {
				grad = make([]float64, len(x))
			}
			model.Score(&GLMParams{x, 1}, grad)
			floats.Scale(-1, grad)
		},
	}

	if model.settings == nil {
		model.settings = &optimize.Settings{}
		model.settings.Recorder = nil
		model.settings.GradientThreshold = 1e-6
	}

	if model.method == nil {
		model.method = &optimize.BFGS{}
	}

	optrslt, err := optimize.Minimize(p, start, model.settings, model.method)
	if err != nil {
		model.failMessage(optrslt)
		panic(err)
	}
	if err = optrslt.Status.Err(); err != nil {
		panic(err)
	}

	params := make([]float64, len(optrslt.X))
	for j := range optrslt.X {
		params[j] = optrslt.X[j]
	}

	fvalue := -optrslt.F

	return params, fvalue
}

// OptSettings allows the caller to provide an optimization settings
// value.
func (model *GLM) OptSettings(s *optimize.Settings) *GLM {
	model.settings = s
	return model
}

// OptMethod sets the optimization method from gonum.Optimize.
func (model *GLM) OptMethod(method optimize.Method) *GLM {
	model.method = method
	return model
}

// failMessage prints information that can help diagnose optimization failures.
func (model *GLM) failMessage(optrslt *optimize.Result) {

	os.Stderr.WriteString("Current point and gradient:\n")
	for j, x := range optrslt.X {
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", x, optrslt.Gradient[j], model.varnames[model.xpos[j]]))
	}

	// Get the mean and standard deviation of covariates.
	mn := make([]float64, len(model.xpos))
	sd := make([]float64, len(model.xpos))
	for j := range model.xpos {
		for i := 0; i < model.NumObs(); i++ {
			mn[j] += float64(model.data[j][i])
		}
		mn[j] /= float64(model.NumObs())
	}
	for j := range model.xpos {
		for i := 0; i < model.NumObs(); i++ {
			u := float64(model.data[j][i]) - mn[j]
			sd[j] += u * u
		}
		sd[j] /= float64(model.NumObs())
	}

	os.Stderr.WriteString("\nCovariate means and standard deviations:\n")
	for j, k := range model.xpos {
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", mn[j], sd[j],
			model.varnames[k]))
	}
}

// EstimateScale returns an estimate of the GLM scale parameter at the
// given parameter values.
func (model *GLM) EstimateScale(params []float64) float64 {

	if model.dispersionMethod == DispersionFixed {
		return model.dispersionValue
	}

	nvar := model.NumParams()
	var ws float64
	var scale float64
	var wgt, off []statmodel.Dtype

	yda := model.data[model.ypos]
	linpred := model.getNslice()
	mn := model.getNslice()
	va := model.getNslice()

	if model.weightpos != -1 {
		wgt = model.data[model.weightpos]
	}
	if model.offsetpos != -1 {
		off = model.data[model.offsetpos]
	}

	for j, k := range model.xpos {
		xda := model.data[k]
		for i := range xda {
			linpred[i] += params[j] * float64(xda[i])
		}
	}
	if off != nil {
		for i := range linpred {
			linpred[i] += float64(off[i])
		}
	}

	// The mean response and variance
	model.link.InvLink(linpred, mn)
	model.vari.Var(mn, va)

	for i := range yda {
		r := float64(yda[i]) - mn[i]
		if wgt == nil {
			scale += r * r / va[i]
			ws += 1
		} else {
			scale += float64(wgt[i]) * r * r / va[i]
			ws += float64(wgt[i])
		}
	}

	scale /= (ws - float64(nvar))

	model.putNslice(linpred)
	model.putNslice(mn)
	model.putNslice(va)

	return scale
}

// resize returns a float64 slice of length n, using the initial
// subslice of x if it is big enough.
func resize(x []float64, n int) []float64 {
	if cap(x) >= n {
		return x[0:n]
	}
	return make([]float64, n)
}

// zero sets all elements of the slice to 0
func zero(x []float64) {
	for i := range x {
		x[i] = 0
	}
}

// zerodtype sets all elements of the slice to 0
func zerodtype(x []statmodel.Dtype) {
	for i := range x {
		x[i] = 0
	}
}

// one sets all elements of the slice to 1
func one(x []float64) {
	for i := range x {
		x[i] = 1
	}
}

// GLMSummary summarizes a fitted generalized linear model.
type GLMSummary struct {

	// The GLM
	model *GLM

	// The results structure
	results *GLMResults

	// Transform the parameters with this function.  If nil,
	// no transformation is applied.  If paramXform is provided,
	// the standard error and Z-score are not shown.
	paramXform func(float64) float64

	// Messages that are appended to the table
	messages []string
}

// SetScale sets the scale on which the parameter results are
// displayed in the summary.  'xf' is a function that maps
// parameters and confidence limits from the linear scale to
// the desired scale.  'msg' is a message that is appended
// to the summary table.
func (gs *GLMSummary) SetScale(xf func(float64) float64, msg string) *GLMSummary {
	gs.paramXform = xf
	gs.messages = append(gs.messages, msg)
	return gs
}

// String returns a string representation of a summary table for the model.
func (gs *GLMSummary) String() string {

	xf := func(x float64) float64 {
		return x
	}

	if gs.paramXform != nil {
		xf = gs.paramXform
	}

	sum := &statmodel.SummaryTable{
		Msg: gs.messages,
	}

	sum.Title = "Generalized linear model analysis"

	sum.Top = []string{
		fmt.Sprintf("Family:   %s", gs.model.fam.Name),
		fmt.Sprintf("Link:     %s", gs.model.link.Name),
		fmt.Sprintf("Variance: %s", gs.model.vari.Name),
		fmt.Sprintf("Num obs:  %d", gs.model.NumObs()),
		fmt.Sprintf("Scale:    %f", gs.results.scale),
	}

	l1 := gs.model.l1wgt != nil

	if !l1 {
		if gs.paramXform == nil {
			sum.ColNames = []string{"Variable   ", "Parameter", "SE", "LCB", "UCB", "Z-score", "P-value"}
		} else {
			sum.ColNames = []string{"Variable   ", "Parameter", "LCB", "UCB", "P-value"}
		}
	} else {
		sum.ColNames = []string{"Variable   ", "Parameter"}
	}

	// String formatter
	fs := func(x interface{}, h string) []string {
		y := x.([]string)
		m := len(h)
		for i := range y {
			if len(y[i]) > m {
				m = len(y[i])
			}
		}
		var z []string
		for i := range y {
			c := fmt.Sprintf("%%-%ds", m)
			z = append(z, fmt.Sprintf(c, y[i]))
		}
		return z
	}

	// Number formatter
	fn := func(x interface{}, h string) []string {
		y := x.([]float64)
		var s []string
		for i := range y {
			s = append(s, fmt.Sprintf("%10.4f", y[i]))
		}
		return s
	}

	if !l1 {
		if gs.paramXform == nil {
			sum.ColFmt = []statmodel.Fmter{fs, fn, fn, fn, fn, fn, fn}
		} else {
			sum.ColFmt = []statmodel.Fmter{fs, fn, fn, fn, fn}
		}
	} else {
		sum.ColFmt = []statmodel.Fmter{fs, fn}
	}

	if !l1 {
		// Create estimate and CI for the parameters
		var par, lcb, ucb []float64
		pax := gs.results.Params()
		for j := range gs.results.Params() {
			par = append(par, xf(pax[j]))
			lcb = append(lcb, xf(pax[j]-2*gs.results.StdErr()[j]))
			ucb = append(ucb, xf(pax[j]+2*gs.results.StdErr()[j]))
		}

		if gs.paramXform == nil {
			sum.Cols = []interface{}{
				gs.results.Names(),
				par,
				gs.results.StdErr(),
				lcb,
				ucb,
				gs.results.ZScores(),
				gs.results.PValues(),
			}
		} else {
			sum.Cols = []interface{}{
				gs.results.Names(),
				par,
				lcb,
				ucb,
				gs.results.PValues(),
			}
		}
	} else {
		sum.Cols = []interface{}{
			gs.results.Names(),
			gs.results.Params(),
		}
	}

	return sum.String()
}

// Summary displays a summary table of the model results.
func (rslt *GLMResults) Summary() *GLMSummary {

	model := rslt.Model().(*GLM)

	return &GLMSummary{
		model:   model,
		results: rslt,
	}
}

// LinearPredictor returns the linear combination of the model covariates based
// on the provided parameter vector.  The provided slice is used if it is large
// enough, otherwise a new slice is allocated.  The linear predictor is returned.
func (model *GLM) LinearPredictor(params *GLMParams, lp []float64) []float64 {

	nobs := model.NumObs()
	coeff := params.coeff

	if cap(lp) < nobs {
		lp = make([]float64, nobs)
	} else {
		lp = lp[0:nobs]
		zero(lp)
	}

	if model.offsetpos != -1 {
		off := model.data[model.offsetpos]
		for i := range lp {
			lp[i] += float64(off[i])
		}
	}

	for j, k := range model.xpos {
		xda := model.data[k]
		for i := 0; i < model.NumObs(); i++ {
			lp[i] += coeff[j] * float64(xda[i])
		}
	}

	return lp
}

// LinearPredictor returns the fitted linear predictor.  If the provided
// slice is large enough, it is used, otherwise a new allocation is made.
// The fitted linear predictor is returned.
func (rslt *GLMResults) LinearPredictor(lp []float64) []float64 {
	model := rslt.Model().(*GLM)
	params := &GLMParams{rslt.Params(), rslt.scale}
	return model.LinearPredictor(params, nil)
}

// Mean returns the fitted mean of the GLM for the given parameter.  If
// the provided slice 'mn' is large enough to hold the result, it is used,
// otherwise a new slice is allocated.  The fitted means are returned.
func (model *GLM) Mean(pa *GLMParams, mn []float64) []float64 {

	mn = model.LinearPredictor(pa, mn)
	model.link.InvLink(mn, mn)

	return mn
}

// Mean returns the fitted mean of the GLM at the estimated parameters.  If
// the provided slice 'mn' is large enough to hold the result, it is used,
// otherwise a new allocation is made. The fitted means are returned.
func (rslt *GLMResults) Mean() []float64 {
	model := rslt.Model().(*GLM)
	params := &GLMParams{rslt.Params(), rslt.scale}
	return model.Mean(params, nil)
}

// Resid returns the residuals (observed minus fitted values) for the model,
// at the given parameter vector.
func (model *GLM) Resid(pa *GLMParams, resid []float64) []float64 {

	resid = model.Mean(pa, resid)

	yda := model.data[model.ypos]
	for i := range yda {
		resid[i] = float64(yda[i]) - resid[i]
	}

	return resid
}

// Resid returns the residuals (observed minus fitted values) at the fitted
// parameter value.
func (rslt *GLMResults) Resid(resid []float64) []float64 {
	model := rslt.Model().(*GLM)
	params := &GLMParams{rslt.Params(), rslt.scale}
	return model.Resid(params, nil)
}

// Variance returns the model-based variance of the GLM responses for the given parameter.  If
// the provided slice is large enough to hold the variances, it is used, otherwise
// a new slice is allocated.  The variances are returned.
func (model *GLM) Variance(pa *GLMParams, va []float64) []float64 {

	va = model.Mean(pa, va)
	model.vari.Var(va, va)
	floats.Scale(pa.scale, va)

	return va
}

// PearsonResid calculates the Pearson residuals at the given parameter value.
// The Pearson residuals are the standardized residuals, using the model standard
// deviation to standardize.  If the provided slice is large enough to hold the
// result, it is used, otherwise a new slice is allocated.  The Pearson standardized
// residuals are returned.
func (model *GLM) PearsonResid(pa *GLMParams, resid []float64) []float64 {

	n := model.NumObs()
	if cap(resid) < n {
		resid = make([]float64, n)
	} else {
		resid = resid[0:n]
		zero(resid)
	}

	mn := model.Mean(pa, nil)
	va := make([]float64, len(mn))
	model.vari.Var(mn, va)
	floats.Scale(pa.scale, va)

	yda := model.data[model.ypos]
	for i := range yda {
		resid[i] = (float64(yda[i]) - mn[i]) / math.Sqrt(va[i])
	}

	return resid
}

// PearsonResid calculates the Pearson residuals at the given parameter value.
// The Pearson residuals are the standardized residuals, using the model standard
// deviation to standardize.  If the provided slice is large enough to hold the
// result, it is used, otherwise a new slice is allocated.  The Pearson standardized
// residuals are returned.
func (rslt *GLMResults) PearsonResid(resid []float64) []float64 {

	model := rslt.Model().(*GLM)
	pa := &GLMParams{rslt.Params(), rslt.scale}
	return model.PearsonResid(pa, resid)
}
