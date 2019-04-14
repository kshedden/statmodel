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

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel/statmodel"
)

// GLM represents a generalized linear model.
type GLM struct {
	data dstream.Dstream

	// Positions of the covariates
	xpos []int

	// Name and position of the outcome variable
	yname string
	ypos  int

	// Name and position of the offset variable, if present.
	offsetname string
	offsetpos  int

	// Name and position of the weight variable, if present.
	weightname string
	weightpos  int

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

	// L1 (lasso) penalty weight.  FitMethod is ignored if
	// non-zero.
	l1wgt []float64

	// L2 (ridge) penalty weights, optional.  Must fit using
	// Gradient method if present.
	l2wgt []float64

	// The L2 norm of every covariate.  If norm=true,
	// calculations are done on normalized covariates.
	xn []float64

	// The internal scaling of the covariates.
	scaletype statmodel.ScaleType

	// Optimization settings
	settings *optimize.Settings

	// Optimization method
	method optimize.Method

	// If not nil, write log messages here
	log *log.Logger

	// Use concurrent calculations in IRLS if the chunk size is at least
	// as large as this value.
	concurrentIRLS int
}

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

// Log takes a Logger value that will be used to log the results of the fit.
func (glm *GLM) Log(log *log.Logger) *GLM {
	glm.log = log
	return glm
}

// NumParams returns the number of covariates in the model.
func (glm *GLM) NumParams() int {
	return len(glm.xpos)
}

// Xpos returns the positions of the covariates in the model's data
// stream.
func (glm *GLM) Xpos() []int {
	return glm.xpos
}

// DataSet returns the data stream that is used to fit the model.
func (glm *GLM) DataSet() dstream.Dstream {
	return glm.data
}

// ConcurrentIRLS sets the minimum chunk size for which concurrent
// calculations are used during IRLS.
func (glm *GLM) ConcurrentIRLS(n int) *GLM {
	glm.concurrentIRLS = n
	return glm
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

// NewGLM creates a new GLM object for the given family, using its
// default link and variance functions.
func NewGLM(data dstream.Dstream, yname string) *GLM {

	return &GLM{
		data:           data,
		yname:          yname,
		fitMethod:      "IRLS",
		concurrentIRLS: 1000,
	}
}

// CovariateScale determines the type of internal scaling of the covariates.
// The default is to do no rescaling of covariates.
func (glm *GLM) CovariateScale(scaletype statmodel.ScaleType) *GLM {
	glm.scaletype = scaletype
	return glm
}

// FitMethod sets the fitting method, either IRLS or gradient.
func (glm *GLM) FitMethod(method string) *GLM {
	lmethod := strings.ToLower(method)
	if lmethod != "irls" && lmethod != "gradient" && lmethod != "coordinate" {
		msg := fmt.Sprintf("GLM fitting method %s not allowed.\n", method)
		panic(msg)
	}
	glm.fitMethod = lmethod
	return glm
}

// Offset sets the name of the offset variable
func (glm *GLM) Offset(name string) *GLM {
	glm.offsetname = name
	return glm
}

// Weight sets the name of the weight variable.
func (glm *GLM) Weight(name string) *GLM {
	glm.weightname = name
	return glm
}

// Family sets the name of the GLM family variable.
func (glm *GLM) Family(fam *Family) *GLM {
	glm.fam = fam
	return glm
}

// L2Weight set the L2 weights used for ridge-regularization.  When
// using L2 weights it is advisable to call ScaleType as well so that
// the weights have equal impacts on the covariates.
func (glm *GLM) L2Weight(l2wgt []float64) *GLM {
	glm.l2wgt = l2wgt
	return glm
}

// L1Weight set the L1 weights used for ridge-regularization.  When
// using L1 weights it is advisable to call ScaleType as well so that
// the weights have equal impacts on the covariates.
func (glm *GLM) L1Weight(l1wgt []float64) *GLM {
	glm.l1wgt = l1wgt
	return glm
}

// Start sets starting values for the fitting algorithm.
func (glm *GLM) Start(start []float64) *GLM {
	glm.start = start
	return glm
}

// Link sets the link function.
func (glm *GLM) Link(link *Link) *GLM {

	if glm.fam == nil {
		panic("Must set family before setting link.\n")
	}
	if !glm.fam.IsValidLink(link) {
		panic("Invalid link")
	}
	glm.link = link

	if strings.ToLower(glm.fam.Name) == "negbinom" {
		// Need to reset the family when the link changes
		glm.fam = NewNegBinomFamily(glm.fam.alpha, link)
	}

	return glm
}

// VarFunc sets the GLM variance function.
func (glm *GLM) VarFunc(va *Variance) *GLM {
	glm.vari = va
	return glm
}

func (glm *GLM) findvars() {

	glm.offsetpos = -1
	glm.weightpos = -1
	glm.ypos = -1
	glm.xpos = glm.xpos[0:0]

	for k, na := range glm.data.Names() {
		switch na {
		case glm.yname:
			glm.ypos = k
		case glm.weightname:
			glm.weightpos = k
		case glm.offsetname:
			glm.offsetpos = k
		default:
			glm.xpos = append(glm.xpos, k)
		}
	}

	if glm.ypos == -1 {
		msg := fmt.Sprintf("Outcome variable '%s' not found.", glm.yname)
		panic(msg)
	}
	if glm.weightpos == -1 && glm.weightname != "" {
		msg := fmt.Sprintf("Weight variable '%s' not found.", glm.weightname)
		panic(msg)
	}
	if glm.offsetpos == -1 && glm.offsetname != "" {
		msg := fmt.Sprintf("Offset variable '%s' not found.", glm.offsetname)
		panic(msg)
	}
}

func (glm *GLM) setup() {

	if glm.link == nil {
		glm.link = NewLink(glm.fam.validLinks[0])
	}

	if glm.vari == nil {
		switch glm.fam.TypeCode {
		case BinomialFamily:
			glm.vari = NewVariance(BinomialVar)
		case PoissonFamily:
			glm.vari = NewVariance(IdentityVar)
		case QuasiPoissonFamily:
			glm.vari = NewVariance(IdentityVar)
		case GaussianFamily:
			glm.vari = NewVariance(ConstantVar)
		case GammaFamily:
			glm.vari = NewVariance(SquaredVar)
		case InvGaussianFamily:
			glm.vari = NewVariance(CubedVar)
		case NegBinomFamily:
			glm.vari = NewNegBinomVariance(glm.fam.alpha)
		default:
			msg := fmt.Sprintf("Unknown GLM family: %s\n", glm.fam.Name)
			panic(msg)
		}
	}
}

func (glm *GLM) check() {

	if glm.l1wgt != nil && len(glm.l1wgt) != len(glm.xpos) {
		msg := fmt.Sprintf("GLM: The L1 weight vector has length %d, but the model has %d covariates.\n",
			len(glm.l1wgt), len(glm.xpos))
		panic(msg)
	}

	if glm.l2wgt != nil && len(glm.l2wgt) != len(glm.xpos) {
		msg := fmt.Sprintf("GLM: The L2 weight vector has length %d, but the model has %d covariates.\n",
			len(glm.l2wgt), len(glm.xpos))
		panic(msg)
	}
}

// Done completes definition of a GLM.  After calling Done the GLM can
// be fit by calling the Fit method.
func (glm *GLM) Done() *GLM {

	if glm.fam == nil {
		msg := "GLM: the family must be defined before calling Done.\n"
		panic(msg)
	}

	glm.findvars()
	glm.doScale()
	glm.setup()

	if len(glm.start) == 0 {
		glm.start = make([]float64, glm.NumParams())
	}

	glm.check()

	return glm
}

// doScale calculates covariate scaling factors.
func (glm *GLM) doScale() {

	glm.xn = make([]float64, len(glm.xpos))

	if glm.scaletype == statmodel.NoScale {
		for k := range glm.xn {
			glm.xn[k] = 1
		}
		return
	}

	// Calculate the L2 norms of the covariates.
	glm.data.Reset()
	var n float64
	for glm.data.Next() {
		for j, k := range glm.xpos {
			x := glm.data.GetPos(k).([]float64)
			if j == 0 {
				n += float64(len(x))
			}
			for i := range x {
				glm.xn[j] += x[i] * x[i]
			}
		}
	}

	for j := range glm.xn {

		// Panic if a covariate has no variation.
		if glm.xn[j] == 0 {
			names := glm.data.Names()
			name := names[glm.xpos[j]]
			msg := fmt.Sprintf("Variable %s has zero variance.\n", name)
			panic(msg)
		}

		switch glm.scaletype {
		case statmodel.L2Norm:
			glm.xn[j] = math.Sqrt(glm.xn[j])
		case statmodel.Variance:
			glm.xn[j] = math.Sqrt(glm.xn[j] / n)
		default:
			panic("unknown scaletype")
		}
	}

	glm.data.Reset()
}

// SetFamily is a convenience method that sets the family, link, and
// variance function based on the given family name.  The link and
// variance functions are set to their canonical values.
func (glm *GLM) SetFamily(fam FamilyType) *GLM {

	switch fam {
	case BinomialFamily:
		glm.fam = &binomial
		glm.link = NewLink(LogitLink)
		glm.vari = NewVariance(BinomialVar)
	case PoissonFamily:
		glm.fam = &poisson
		glm.link = NewLink(LogLink)
		glm.vari = NewVariance(IdentityVar)
	case QuasiPoissonFamily:
		glm.fam = &quasiPoisson
		glm.link = NewLink(LogLink)
		glm.vari = NewVariance(IdentityVar)
	case GaussianFamily:
		glm.fam = &gaussian
		glm.link = NewLink(IdentityLink)
		glm.vari = NewVariance(ConstantVar)
	case GammaFamily:
		glm.fam = &gamma
		glm.link = NewLink(RecipLink)
		glm.vari = NewVariance(SquaredVar)
	case InvGaussianFamily:
		glm.fam = &invGaussian
		glm.link = NewLink(RecipSquaredLink)
		glm.vari = NewVariance(CubedVar)
	case NegBinomFamily:
		panic("GLM: can't set family to NegBinom using SetFamily")
	default:
		msg := fmt.Sprintf("Unknown GLM family: %v\n", fam)
		panic(msg)
	}

	return glm
}

// LogLike returns the log-likelihood value for the generalized linear
// model at the given parameter values.
func (glm *GLM) LogLike(params statmodel.Parameter) float64 {

	gpar := params.(*GLMParams)
	coeff := gpar.coeff
	scale := gpar.scale

	var loglike float64
	var linpred []float64
	var mn []float64

	glm.data.Reset()

	for glm.data.Next() {

		var yda, wgts, off []float64

		yda = glm.data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgts = glm.data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.data.GetPos(glm.offsetpos).([]float64)
		}

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)

		// Update the linear predictor
		zero(linpred)
		for j, k := range glm.xpos {
			xda := glm.data.GetPos(k).([]float64)
			floats.AddScaled(linpred, coeff[j]/glm.xn[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		// Update the log likelihood value
		glm.link.InvLink(linpred, mn)
		loglike += glm.fam.LogLike(yda, mn, wgts, scale)
	}

	// Account for the L2 penalty
	if glm.l2wgt != nil {
		nobs := float64(glm.data.NumObs())
		for j, v := range glm.l2wgt {
			loglike -= nobs * v * coeff[j] * coeff[j] / 2
		}
	}

	return loglike
}

func scoreFactor(yda, mn, deriv, va, sfac []float64) {
	for i, y := range yda {
		sfac[i] = (y - mn[i]) / (deriv[i] * va[i])
	}
}

// Score returns the score vector for the generalized linear model at
// the given parameter values.
func (glm *GLM) Score(params statmodel.Parameter, score []float64) {

	gpar := params.(*GLMParams)
	coeff := gpar.coeff

	var linpred, mn, deriv, va, fac, facw []float64

	glm.data.Reset()
	zero(score)

	for glm.data.Next() {

		var yda, wgts, off []float64

		yda = glm.data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgts = glm.data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.data.GetPos(glm.offsetpos).([]float64)
		}

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		deriv = resize(deriv, n)
		va = resize(va, n)
		fac = resize(fac, n)
		facw = resize(facw, n)

		// Update the linear predictor
		zero(linpred)
		for j, k := range glm.xpos {
			xda := glm.data.GetPos(k).([]float64)
			floats.AddScaled(linpred, coeff[j]/glm.xn[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		glm.link.InvLink(linpred, mn)
		glm.link.Deriv(mn, deriv)
		glm.vari.Var(mn, va)

		scoreFactor(yda, mn, deriv, va, fac)

		for j, k := range glm.xpos {

			xda := glm.data.GetPos(k).([]float64)

			if wgts == nil {
				score[j] += floats.Dot(fac, xda) / glm.xn[j]
			} else {
				floats.MulTo(facw, fac, wgts)
				score[j] += floats.Dot(facw, xda) / glm.xn[j]
			}
		}
	}

	// Account for the L2 penalty
	if glm.l2wgt != nil {
		nobs := float64(glm.data.NumObs())
		for j, v := range glm.l2wgt {
			score[j] -= nobs * v * coeff[j]
		}
	}
}

// Hessian returns the Hessian matrix for the model.  The Hessian is
// returned as a one-dimensional array, which is the vectorized form
// of the Hessian matrix.  Either the observed or expected Hessian can
// be calculated.
func (glm *GLM) Hessian(param statmodel.Parameter, ht statmodel.HessType, hess []float64) {

	gpar := param.(*GLMParams)
	coeff := gpar.coeff

	var linpred, mn, lderiv, lderiv2, va, vad, fac, sfac []float64

	nvar := glm.NumParams()
	xdat := make([][]float64, nvar)
	glm.data.Reset()
	zero(hess)

	for glm.data.Next() {

		var yda, wgts, off []float64

		for j, k := range glm.xpos {
			xdat[j] = glm.data.GetPos(k).([]float64)
		}

		yda = glm.data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgts = glm.data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.data.GetPos(glm.offsetpos).([]float64)
		}

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		lderiv = resize(lderiv, n)
		va = resize(va, n)
		fac = resize(fac, n)
		sfac = resize(sfac, n)

		// Update the linear predictor
		zero(linpred)
		for j := range glm.xpos {
			floats.AddScaled(linpred, coeff[j], xdat[j])
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		// The mean response
		glm.link.InvLink(linpred, mn)

		glm.link.Deriv(mn, lderiv)
		glm.vari.Var(mn, va)

		// Factor for the expected Hessian
		for i := 0; i < len(lderiv); i++ {
			fac[i] = 1 / (lderiv[i] * lderiv[i] * va[i])
		}

		// Adjust the factor for the observed Hessian
		if ht == statmodel.ObsHess {
			vad = resize(vad, n)
			lderiv2 = resize(lderiv2, n)
			glm.link.Deriv2(mn, lderiv2)
			glm.vari.Deriv(mn, vad)
			scoreFactor(yda, mn, lderiv, va, sfac)

			for i := range fac {
				h := va[i]*lderiv2[i] + lderiv[i]*vad[i]
				h *= sfac[i] * fac[i]
				if wgts != nil {
					h *= wgts[i]
				}
				fac[i] *= 1 + h
			}
		}

		// Update the Hessian matrix
		glm.hessXprod(xdat, fac, wgts, hess)
	}

	// Fill in the upper triangle
	for j1 := range glm.xpos {
		for j2 := 0; j2 < j1; j2++ {
			hess[j2*nvar+j1] = hess[j1*nvar+j2]
		}
	}

	// Account for the L2 penalty
	if glm.l2wgt != nil {
		nobs := float64(glm.data.NumObs())
		for j, v := range glm.l2wgt {
			hess[j*nvar+j] -= nobs * v
		}
	}
}

func (glm *GLM) hessXprod(xdat [][]float64, fac, wgts, hess []float64) {

	nvar := len(xdat)

	var wg sync.WaitGroup

	for j1 := range glm.xpos {
		for j2 := 0; j2 <= j1; j2++ {

			wg.Add(1)
			go func(j1, j2 int) {
				x1 := xdat[j1]
				x2 := xdat[j2]
				if wgts == nil {
					for i := range x1 {
						hess[j1*nvar+j2] -= fac[i] * x1[i] * x2[i]
					}
				} else {
					for i := range x1 {
						hess[j1*nvar+j2] -= wgts[i] * fac[i] * x1[i] * x2[i]
					}
				}
				wg.Done()
			}(j1, j2)
		}
	}

	wg.Wait()
}

// GetFocusable returns a focusable version of the model.
// This is exposed for use in elastic net optimization.
func (glm *GLM) GetFocusable() statmodel.ModelFocuser {

	other := []string{glm.yname}
	if glm.weightpos != -1 {
		other = append(other, glm.weightname)
	}

	// Set up the focusable data.
	fdat := statmodel.NewFocusData(glm.data, glm.xpos, glm.xn).Other(other)
	if glm.offsetpos != -1 {
		// An actual offset, to be combined if present with
		// the offset that results by combining the non-focus
		// covariates.
		fdat.Offset(glm.offsetpos)
	}
	fdat = fdat.Done()

	newglm := NewGLM(fdat, glm.yname).Family(glm.fam).Link(glm.link).VarFunc(glm.vari).Offset("off")
	if glm.weightpos != -1 {
		newglm = newglm.Weight(glm.weightname)
	} else if glm.weightname != "" {
		newglm = newglm.Weight(glm.weightname)
	}

	if glm.l1wgt != nil {
		newglm = newglm.L1Weight(make([]float64, 1)) //g.l1wgt)
	}

	if glm.l2wgt != nil {
		newglm = newglm.L2Weight(make([]float64, 1)) //g.l2wgt)
	}

	newglm.Done()

	return newglm
}

// Focus sets the data to contain only one predictor (with the given
// index).  The effects of the remaining covariates are captured
// through dthe offset.  The method is exposed for use in elastic net
// fitting, but is unlikely to be useful for ordinary users.  Can only be
// called on a focusable version of the model value.
func (glm *GLM) Focus(j int, coeff []float64, l2wgt float64) {

	glm.data.(*statmodel.FocusData).Focus(j, coeff)

	if l2wgt > 0 {
		glm.l2wgt[0] = l2wgt
	}
}

// fitRegularized estimates the parameters of the GLM using L1
// regularization (with optimal L2 regularization).  This invokes
// coordinate descent optimization.  For fitting with no L1
// regularization (with or without L2 regularization), call
// fitGradient which invokes gradient optimization.
func (glm *GLM) fitRegularized() *GLMResults {

	if glm.log != nil {
		log.Print("Regularized fitting\n")
	}

	start := &GLMParams{
		coeff: make([]float64, len(glm.xpos)),
		scale: 1.0,
	}

	checkstep := strings.ToLower(glm.fam.Name) != "gaussian"
	par := statmodel.FitL1Reg(glm, start, glm.l1wgt, glm.l2wgt, glm.xn, checkstep)
	coeff := par.GetCoeff()

	// Since coeff is transformed back to the original scale, we
	// need to stop normalizing (else EstimateScale and other
	// post-fit quantities will be wrong).
	for i := range glm.xn {
		glm.xn[i] = 1
	}

	// Covariate names
	var xna []string
	na := glm.data.Names()
	for _, j := range glm.xpos {
		xna = append(xna, na[j])
	}

	scale := glm.EstimateScale(coeff)

	results := &GLMResults{
		BaseResults: statmodel.NewBaseResults(glm, 0, coeff, xna, nil),
		scale:       scale,
	}

	return results
}

// Fit estimates the parameters of the GLM and returns a results
// object.  Unregularized fits and fits involving L2 regularization
// can be obtained, but if L1 regularization is desired use
// FitRegularized instead of Fit.
func (glm *GLM) Fit() *GLMResults {

	if glm.l1wgt != nil {
		return glm.fitRegularized()
	}

	nvar := glm.NumParams()
	maxiter := 20

	var start []float64
	if glm.start != nil {
		start = glm.start
	} else {
		start = make([]float64, nvar)
	}

	if glm.l2wgt != nil {
		glm.fitMethod = "gradient"
	}

	var params []float64

	if strings.ToLower(glm.fitMethod) == "gradient" {
		if glm.log != nil {
			log.Print("Unregularized fitting using gradient optimization\n")
		}
		params, _ = glm.fitGradient(start)
	} else {
		if glm.log != nil {
			log.Print("Unregularized fitting using IRLS\n")
		}
		params = glm.fitIRLS(start, maxiter)
	}

	// Everything remaining does not use scaling
	for j := range glm.xn {
		glm.xn[j] = 1
	}

	scale := glm.EstimateScale(params)

	vcov, _ := statmodel.GetVcov(glm, &GLMParams{params, scale})
	floats.Scale(scale, vcov)

	ll := glm.LogLike(&GLMParams{params, scale})

	var xna []string
	na := glm.data.Names()
	for _, j := range glm.xpos {
		xna = append(xna, na[j])
	}

	results := &GLMResults{
		BaseResults: statmodel.NewBaseResults(glm, ll, params, xna, vcov),
		scale:       scale,
	}

	return results
}

// fitGradient uses gradient-based optimization to obtain the fitted
// GLM parameters.
func (glm *GLM) fitGradient(start []float64) ([]float64, float64) {

	p := optimize.Problem{
		Func: func(x []float64) float64 {
			return -glm.LogLike(&GLMParams{x, 1})
		},
		Grad: func(grad, x []float64) []float64 {
			if len(grad) != len(x) {
				grad = make([]float64, len(x))
			}
			glm.Score(&GLMParams{x, 1}, grad)
			floats.Scale(-1, grad)
			return grad
		},
	}

	if glm.settings == nil {
		glm.settings = &optimize.Settings{}
		glm.settings.Recorder = nil
		glm.settings.GradientThreshold = 1e-6
	}

	if glm.method == nil {
		glm.method = &optimize.BFGS{}
	}

	optrslt, err := optimize.Minimize(p, start, glm.settings, glm.method)
	if err != nil {
		glm.failMessage(optrslt)
		panic(err)
	}
	if err = optrslt.Status.Err(); err != nil {
		panic(err)
	}

	params := make([]float64, len(optrslt.X))
	for j := range optrslt.X {
		params[j] = optrslt.X[j] / glm.xn[j]
	}

	fvalue := -optrslt.F

	return params, fvalue
}

// OptSettings allows the caller to provide an optimization settings
// value.
func (glm *GLM) OptSettings(s *optimize.Settings) *GLM {
	glm.settings = s
	return glm
}

// OptMethod sets the optimization method from gonum.Optimize.
func (glm *GLM) OptMethod(method optimize.Method) *GLM {
	glm.method = method
	return glm
}

// failMessage prints information that can help diagnose optimization failures.
func (glm *GLM) failMessage(optrslt *optimize.Result) {

	xnames := glm.data.Names()

	os.Stderr.WriteString("Current point and gradient:\n")
	for j, x := range optrslt.X {
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", x, optrslt.Gradient[j], xnames[glm.xpos[j]]))
	}

	// Get the covariates to avoid repeated type assertions
	glm.data.Reset()
	xvars := make([][]float64, len(glm.xpos))
	for glm.data.Next() {
		for k, j := range glm.xpos {
			xvars[k] = append(xvars[k], glm.data.GetPos(j).([]float64)...)
		}
	}

	// Get the mean and standard deviation of covariates.
	mn := make([]float64, len(glm.xpos))
	sd := make([]float64, len(glm.xpos))
	for j, x := range xvars {
		mn[j] = floats.Sum(x) / float64(len(x))
	}
	for j, x := range xvars {
		for _, y := range x {
			u := y - mn[j]
			sd[j] += u * u
		}
		sd[j] /= float64(len(x))
		sd[j] = math.Sqrt(sd[j])
	}

	os.Stderr.WriteString("\nCovariate means and standard deviations:\n")
	for j, m := range mn {
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", m, sd[j], xnames[glm.xpos[j]]))
	}
}

// EstimateScale returns an estimate of the GLM scale parameter at the
// given parameter values.
func (glm *GLM) EstimateScale(params []float64) float64 {

	name := strings.ToLower(glm.fam.Name)
	if name == "binomial" || name == "poisson" {
		return 1
	}

	nvar := glm.NumParams()
	var linpred []float64
	var mn []float64
	var va []float64
	var ws float64

	glm.data.Reset()
	var scale float64
	for glm.data.Next() {

		var yda, wgt, off []float64

		yda = glm.data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgt = glm.data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.data.GetPos(glm.offsetpos).([]float64)
		}

		linpred = resize(linpred, n)
		mn = resize(mn, n)
		va = resize(va, n)

		zero(linpred)
		for j, k := range glm.xpos {
			xda := glm.data.GetPos(k).([]float64)
			for i, x := range xda {
				linpred[i] += params[j] * x
			}
		}
		if off != nil {
			floats.AddTo(linpred, linpred, off)
		}

		// The mean response and variance
		glm.link.InvLink(linpred, mn)
		glm.vari.Var(mn, va)

		for i, y := range yda {
			r := y - mn[i]
			if wgt == nil {
				scale += r * r / va[i]
				ws += 1
			} else {
				scale += wgt[i] * r * r / va[i]
				ws += wgt[i]
			}
		}
	}

	scale /= (ws - float64(nvar))

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

// one sets all elements of the slice to 1
func one(x []float64) {
	for i := range x {
		x[i] = 1
	}
}

// GLMSummary summarizes a fitted generalized linear model.
type GLMSummary struct {

	// The GLM
	glm *GLM

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
		fmt.Sprintf("Family:   %s", gs.glm.fam.Name),
		fmt.Sprintf("Link:     %s", gs.glm.link.Name),
		fmt.Sprintf("Variance: %s", gs.glm.vari.Name),
		fmt.Sprintf("Num obs:  %d", gs.glm.DataSet().NumObs()),
		fmt.Sprintf("Scale:    %f", gs.results.scale),
	}

	l1 := gs.glm.l1wgt != nil

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

	glm := rslt.Model().(*GLM)

	return &GLMSummary{
		glm:     glm,
		results: rslt,
	}
}
