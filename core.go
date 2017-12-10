package statmodel

import (
	"bytes"
	"fmt"
	"math"
	"strings"

	"github.com/gonum/matrix/mat64"
	"github.com/kshedden/dstream/dstream"
)

type HessType int

const (
	ObsHess = iota
	ExpHess
)

// Parameter is the parameter of a model.
type Parameter interface{}

// RegFitter is a regression model that can be fit to data.
type RegFitter interface {

	// Number of parameters in the model.
	NumParams() int

	// Positions of the covariates in the Dstream
	Xpos() []int

	// The dataset, including covariates and outcomes, and if
	// relevant, weights, strata, and other variables.
	DataSet() dstream.Dstream

	// The log-likelihood function
	LogLike(Parameter) float64

	// The score vector
	Score(Parameter, []float64)

	// The Hessian matrix
	Hessian(Parameter, HessType, []float64)
}

type BaseResultser interface {
	Model() RegFitter
	Names() []string
	LogLike() float64
	Params() []float64
	VCov() []float64
	StdErr() []float64
	ZScores() []float64
	PValues() []float64
}

type BaseResults struct {
	model   RegFitter
	loglike float64
	params  []float64
	xnames  []string
	vcov    []float64
	stderr  []float64
	zscores []float64
	pvalues []float64
}

func NewBaseResults(model RegFitter, loglike float64, params []float64, xnames []string, vcov []float64) BaseResults {
	return BaseResults{
		model:   model,
		loglike: loglike,
		params:  params,
		xnames:  xnames,
		vcov:    vcov,
	}
}

func (rslt *BaseResults) Model() RegFitter {
	return rslt.model
}

// FittedValues returns the fitted linear predictor for a regression
// model.  If da is nil, the fitted values are based on the data used
// to it the mode.  If da is provided it is used to produce the fitted
// values, so must have the same columns as the training data.
func (rslt *BaseResults) FittedValues(da dstream.Dstream) []float64 {

	if da == nil {
		da = rslt.model.DataSet()
	}
	fv := make([]float64, da.NumObs())
	ii := 0
	n := 0

	xp := rslt.model.Xpos()

	da.Reset()
	for da.Next() {
		for k, j := range xp {
			z := da.GetPos(j).([]float64)
			n = len(z)
			for i, v := range z {
				fv[ii+i] += rslt.params[k] * v
			}
		}
		ii += n
	}

	return fv
}

func (rslt *BaseResults) Names() []string {
	return rslt.xnames
}

func (rslt *BaseResults) Params() []float64 {
	return rslt.params
}

func (rslt *BaseResults) VCov() []float64 {
	return rslt.vcov
}

func (rslt *BaseResults) LogLike() float64 {
	return rslt.loglike
}

func (rslt *BaseResults) StdErr() []float64 {

	// No vcov, no standard error
	if rslt.vcov == nil {
		return nil
	}

	p := rslt.model.NumParams()
	if rslt.stderr == nil {
		rslt.stderr = make([]float64, p)
	} else {
		return rslt.stderr
	}

	for i, _ := range rslt.stderr {
		rslt.stderr[i] = math.Sqrt(rslt.vcov[i*p+i])
	}

	return rslt.stderr
}

func (rslt *BaseResults) ZScores() []float64 {

	// No vcov, no z-scores
	if rslt.vcov == nil {
		return nil
	}

	p := rslt.model.NumParams()
	if rslt.zscores == nil {
		rslt.zscores = make([]float64, p)
	} else {
		return rslt.zscores
	}

	std := rslt.StdErr()
	for i, _ := range std {
		rslt.zscores[i] = rslt.params[i] / std[i]
	}

	return rslt.zscores
}

func normcdf(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt(2))
}

func (rslt *BaseResults) PValues() []float64 {

	// No vcov, no p-values
	if rslt.vcov == nil {
		return nil
	}

	p := rslt.model.NumParams()
	if rslt.pvalues == nil {
		rslt.pvalues = make([]float64, p)
	} else {
		return rslt.pvalues
	}

	for i, z := range rslt.zscores {
		rslt.pvalues[i] = 2 * normcdf(-math.Abs(z))
	}

	return rslt.pvalues
}

func negative(x []float64) {
	for i := 0; i < len(x); i++ {
		x[i] *= -1
	}
}

func GetVcov(model RegFitter, params Parameter) ([]float64, error) {
	nvar := model.NumParams()
	n2 := nvar * nvar
	hess := make([]float64, n2)
	model.Hessian(params, ExpHess, hess)
	hmat := mat64.NewDense(nvar, nvar, hess)
	hessi := make([]float64, n2)
	himat := mat64.NewDense(nvar, nvar, hessi)
	err := himat.Inverse(hmat)
	if err != nil {
		return nil, err
	}
	himat.Scale(-1, himat)

	return hessi, nil
}

// Summary returns a string that holds a table of coefficients,
// standard errors, z-scores, and p-values for the fitted model.
func (rslt *BaseResults) Summary() string {

	if rslt.params == nil {
		return ""
	}

	p := len(rslt.params)

	tw := 72

	if rslt.xnames == nil {
		rslt.xnames = make([]string, p)
		for k, _ := range rslt.params {
			rslt.xnames[k] = fmt.Sprintf("V%d", k)
		}
	}

	gapw := 2
	gap := strings.Repeat(" ", gapw)

	namesf := fmtstrings(rslt.xnames, 8)
	m := maxlen(namesf)
	w := []int{m, 12, 12, 12, 12}

	paramsf := fmtfloats(rslt.params, p, w[1])

	// May or may not be available
	var stdf, zscoref, pvaluesf []string
	if rslt.StdErr() != nil {
		stdf = fmtfloats(rslt.StdErr(), p, w[2])
		zscoref = fmtfloats(rslt.ZScores(), p, w[3])
		pvaluesf = fmtfloats(rslt.PValues(), p, w[4])
	} else {
		for _, _ = range rslt.params {
			stdf = append(stdf, "")
			zscoref = append(zscoref, "")
			pvaluesf = append(pvaluesf, "")
		}
	}

	var buf bytes.Buffer

	buf.Write([]byte(strings.Repeat("-", tw)))
	buf.Write([]byte("\n"))

	// Header
	c := fmt.Sprintf("%%%ds", w[0]+gapw)
	buf.Write([]byte(fmt.Sprintf(c, "Variable"+gap)))
	c = fmt.Sprintf("%%%ds", w[1]+gapw)
	buf.Write([]byte(fmt.Sprintf(c, "Coefficient"+gap)))
	c = fmt.Sprintf("%%%ds", w[2]+gapw)
	buf.Write([]byte(fmt.Sprintf(c, "StdErr"+gap)))
	c = fmt.Sprintf("%%%ds", w[3]+gapw)
	buf.Write([]byte(fmt.Sprintf(c, "Z-Score"+gap)))
	c = fmt.Sprintf("%%%ds", w[4]+gapw)
	buf.Write([]byte(fmt.Sprintf(c, "P-Value"+gap)))
	buf.Write([]byte("\n"))
	buf.Write([]byte(strings.Repeat("-", tw)))
	buf.Write([]byte("\n"))

	// Parameter information
	for j, na := range namesf {
		buf.Write([]byte(na))
		buf.Write([]byte(gap))
		buf.Write([]byte(paramsf[j]))
		buf.Write([]byte(gap))
		buf.Write([]byte(stdf[j]))
		buf.Write([]byte(gap))
		buf.Write([]byte(zscoref[j]))
		buf.Write([]byte(gap))
		buf.Write([]byte(pvaluesf[j]))
		buf.Write([]byte("\n"))
	}

	buf.Write([]byte(strings.Repeat("-", tw)))
	buf.Write([]byte("\n"))

	return buf.String()
}

func fmtstrings(f []string, minw int) []string {

	w := maxlen(f)
	if w < minw {
		w = minw
	}
	s := make([]string, len(f))
	c := fmt.Sprintf("%%-%ds", w)

	for i, x := range f {
		s[i] = fmt.Sprintf(c, x)
	}

	return s
}

func fmtfloats(f []float64, p int, w int) []string {
	var s []string
	if f != nil {
		s = make([]string, len(f))
	} else {
		return make([]string, p)
	}
	c := fmt.Sprintf("%%%d.4f", w)
	for i, v := range f {
		s[i] = fmt.Sprintf(c, v)
	}
	return s
}

func maxlen(s []string) int {
	k := 0
	for _, v := range s {
		if len(v) > k {
			k = len(v)
		}
	}
	return k
}
