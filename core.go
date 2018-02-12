package statmodel

import (
	"bytes"
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/mat"

	"github.com/kshedden/dstream/dstream"
)

type HessType int

const (
	ObsHess = iota
	ExpHess
)

// Parameter is the parameter of a model.
type Parameter interface {

	// Get the coefficients of the covariates in the linear
	// predictor.  The returned value should be a reference so
	// that changes to it lead to corresponding changes in the
	// parameter itself.
	GetCoeff() []float64

	// Set the coefficients of the covariates in the linear
	// predictor.
	SetCoeff([]float64)

	Clone() Parameter
}

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
	hmat := mat.NewDense(nvar, nvar, hess)
	hessi := make([]float64, n2)
	himat := mat.NewDense(nvar, nvar, hessi)
	err := himat.Inverse(hmat)
	if err != nil {
		return nil, err
	}
	himat.Scale(-1, himat)

	return hessi, nil
}

type Summary struct {

	// Title
	Title string

	// Column names
	ColNames []string

	// Formatters for the column values
	ColFmt []Fmter

	// Cols[j] is the j^th column.  It's concrete type should
	// be an array, e.g. of numbers or strings.
	Cols []interface{}

	// Values at the top of the summary
	Top []string

	// Messages displayed below the table
	Msg []string

	// Total width of the table
	tw int
}

// Draw a line constructed of the given character filling the width of
// the table.
func (s *Summary) line(c string) string {
	return strings.Repeat(c, s.tw) + "\n"
}

// cleanTop ensures that all fields in the top part of the table have
// the same width.
func (s *Summary) cleanTop() {

	w := len(s.Top[0])
	for _, x := range s.Top {
		if len(x) > w {
			w = len(x)
		}
	}

	for i, x := range s.Top {
		if len(x) < w {
			s.Top[i] = x + strings.Repeat(" ", w-len(x))
		}
	}
}

// Construct the upper part of the table, which contains summary
// values for the model.
func (s *Summary) top(gap int) string {

	w := []int{0, 0}

	for j, x := range s.Top {
		if len(x) > w[j%2] {
			w[j%2] = len(x)
		}
	}

	var b bytes.Buffer

	for j, x := range s.Top {
		c := fmt.Sprintf("%%-%ds", w[j%2])
		b.Write([]byte(fmt.Sprintf(c, x)))
		if j%2 == 1 {
			b.Write([]byte("\n"))
		} else {
			b.Write([]byte(strings.Repeat(" ", gap)))
		}
	}

	if len(s.Top)%2 == 1 {
		b.Write([]byte("\n"))
	}

	return b.String()
}

// Fmter formats the elements of an array of values.
type Fmter func(interface{}, string) []string

// String returns the table as a string.
func (s *Summary) String() string {

	s.cleanTop()

	var tab [][]string
	var wx []int
	for j, c := range s.Cols {
		u := s.ColFmt[j](c, s.ColNames[j])
		tab = append(tab, u)
		if len(u[0]) > len(s.ColNames[j]) {
			wx = append(wx, len(u[0]))
		} else {
			wx = append(wx, len(s.ColNames[j]))
		}
	}

	gap := 10

	// Get the total width of the table
	s.tw = 0
	for _, w := range wx {
		s.tw += w
	}
	if s.tw < len(s.Title) {
		s.tw = len(s.Title)
	}
	if s.tw < gap+2*len(s.Top[0]) {
		s.tw = gap + 2*len(s.Top[0])
	}

	var buf bytes.Buffer

	// Center the title
	k := len(s.Title)
	kr := (s.tw - k) / 2
	if kr < 0 {
		kr = 0
	}
	buf.Write([]byte(strings.Repeat(" ", kr)))
	buf.Write([]byte(s.Title))
	buf.Write([]byte("\n"))

	buf.Write([]byte(s.line("=")))
	buf.Write([]byte(s.top(gap)))
	buf.Write([]byte(s.line("-")))

	for j, c := range s.ColNames {
		f := fmt.Sprintf("%%%ds", wx[j])
		buf.Write([]byte(fmt.Sprintf(f, c)))
	}
	buf.Write([]byte("\n"))
	buf.Write([]byte(s.line("-")))

	for i := 0; i < len(tab[0]); i++ {
		for j := 0; j < len(tab); j++ {
			f := fmt.Sprintf("%%%ds", wx[j])
			buf.Write([]byte(fmt.Sprintf(f, tab[j][i])))
		}
		buf.Write([]byte("\n"))
	}
	buf.Write([]byte(s.line("-")))

	if len(s.Msg) > 0 {
		for _, msg := range s.Msg {
			buf.Write([]byte(msg + "\n"))
		}
	}

	return buf.String()
}
