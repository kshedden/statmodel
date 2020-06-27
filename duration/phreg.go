// Package duration supports various methods for statistical analysis
// of duration data (survival analysis).
package duration

import (
	"fmt"
	"log"
	"math"
	"os"
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/optimize"

	"github.com/kshedden/statmodel/statmodel"
)

// PHParameter contains a parameter value for a proportional hazards
// regression model.
type PHParameter struct {
	coeff []float64
}

// GetCoeff returns the array of model coefficients from a parameter value.
func (p *PHParameter) GetCoeff() []float64 {
	return p.coeff
}

// SetCoeff sets the array of model coefficients for a parameter value.
func (p *PHParameter) SetCoeff(x []float64) {
	p.coeff = x
}

// Clone returns a deep copy of the parameter value.
func (p *PHParameter) Clone() statmodel.Parameter {
	q := make([]float64, len(p.coeff))
	copy(q, p.coeff)
	return &PHParameter{q}
}

// PHReg describes a proportional hazards regression model for right
// censored data.
type PHReg struct {

	// The names of the variables.  The order agrees with the order of 'data'.
	varnames []string

	// The data to which the model is fit
	data [][]statmodel.Dtype

	// Starting values, optional
	start []float64

	// Name and position of the event variable
	statuspos int

	// Name and position of the time variable
	timepos int

	// Name and position of the entry time variable
	entrypos int

	// Name and position of an offset variable
	offsetpos int

	// Name and position of a case weight variable
	weightpos int

	// Name and position of a stratum variable
	stratapos int

	// Start and end position of the strata
	stratumix [][2]int

	// The sorted times at which events occur in each stratum
	etimes [][]float64

	// enter[i][j] are the row indices that enter the risk set at
	// the jth distinct time in stratum i
	enter [][][]int

	// event[i][j] are the row indices that have an event at
	// the jth distinct time in stratum i
	event [][][]int

	// exit[i][j] are the row indices that exit the risk set at
	// the jth distinct time in stratum i
	exit [][][]int

	// The sum of covariates with events in each stratum
	sumx [][]float64

	// L2 (ridge) weights for each variable
	l2wgtMap map[string]float64
	l2wgt    []float64

	// L1 (lasso) weights for each variable
	l1wgtMap map[string]float64
	l1wgt    []float64

	// The positions of the covariates in the Dstream
	xpos []int

	// If skip[i] is true, case i is skipped since it is censored before the first event.
	skip []bool

	// The number of cases that are skipped because they are censored before the first event
	skipEarlyCensor int

	// Optimization settings
	optsettings *optimize.Settings

	// Optimization method
	optmethod optimize.Method

	log *log.Logger

	nslices [][]float64
}

// NumObs returns the number of observations in the data set.
func (ph *PHReg) NumObs() int {
	return len(ph.data[0])
}

// NumParams returns the number of model parameters (regression coefficients).
func (ph *PHReg) NumParams() int {
	return len(ph.xpos)
}

// Dataset returns the data columns that are used to fit the model.
func (ph *PHReg) Dataset() [][]statmodel.Dtype {
	return ph.data
}

// Xpos return the positions of the covariates in the model's dstream.
func (ph *PHReg) Xpos() []int {
	return ph.xpos
}

// PHRegConfig defines configuration parameters for a proportional hazards regression..
type PHRegConfig struct {

	// A logger to which logging information is wreitten
	Log *log.Logger

	// Start contains starting values for the regression parameter estimates
	Start []float64

	// WeightVar is the name of the variable for frequency-weighting the cases, if an empty
	// string, all weights are equal to 1.
	WeightVar string

	// OffsetVar is the name of a variable that defines an offset.
	OffsetVar string

	// StrataVar is the name of a variable that defines strata.
	StrataVar string

	// EntryVar is the name of a variable that defines entry (left truncation) times.
	EntryVar string

	L1Penalty map[string]float64
	L2Penalty map[string]float64

	// OptMethod is the Gonum optimization used to fit the model.
	OptMethod optimize.Method

	// OptSettings configures the Gonum optimization routine.
	OptSettings *optimize.Settings
}

// DefaultPHRegConfig returns a default configuration struct for a proportional hazards regression.
func DefaultPHRegConfig() *PHRegConfig {

	return &PHRegConfig{
		OptMethod: &optimize.BFGS{
			Linesearcher: &optimize.MoreThuente{},
		},
	}
}

// zerodtype sets all elements of the slice to 0
func zerodtype(x []statmodel.Dtype) {
	for i := range x {
		x[i] = 0
	}
}

// NewPHReg returns a PHReg value that can be used to fit a
// proportional hazards regression model.
func NewPHReg(data statmodel.Dataset, time, status string, predictors []string, config *PHRegConfig) (*PHReg, error) {

	if config == nil {
		config = DefaultPHRegConfig()
	}

	pos := make(map[string]int)
	for i, v := range data.Names() {
		pos[v] = i
	}

	timepos, ok := pos[time]
	if !ok {
		msg := fmt.Sprintf("Time variable '%s' not found in dataset\n", time)
		return nil, fmt.Errorf(msg)
	}

	statuspos, ok := pos[status]
	if !ok {
		msg := fmt.Sprintf("Status variable '%s' not found in dataset\n", status)
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

	getpos := func(vn string) int {
		if vn == "" {
			return -1
		}
		var loc int
		var ok bool
		loc, ok = pos[vn]
		if !ok {
			msg := fmt.Sprintf("'%s' not found\n", vn)
			panic(msg)
		}
		return loc
	}

	weightpos := getpos(config.WeightVar)
	stratapos := getpos(config.StrataVar)
	offsetpos := getpos(config.OffsetVar)
	entrypos := getpos(config.EntryVar)

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

	ph := &PHReg{
		data:        data.Data(),
		varnames:    varnames,
		timepos:     timepos,
		statuspos:   statuspos,
		xpos:        xpos,
		weightpos:   weightpos,
		offsetpos:   offsetpos,
		entrypos:    entrypos,
		stratapos:   stratapos,
		start:       config.Start,
		l1wgt:       penToSlice(config.L1Penalty),
		l2wgt:       penToSlice(config.L2Penalty),
		l1wgtMap:    config.L1Penalty,
		l2wgtMap:    config.L2Penalty,
		log:         config.Log,
		optsettings: config.OptSettings,
		optmethod:   config.OptMethod,
	}

	ph.init()

	return ph, nil
}

func (ph *PHReg) init() {
	ph.sortByStratum()
	ph.setupTimes()
	ph.setupCovs()
}

func (a argsort) Len() int {
	return len(a.s)
}

func (a argsort) Swap(i, j int) {
	a.s[i], a.s[j] = a.s[j], a.s[i]
	a.inds[i], a.inds[j] = a.inds[j], a.inds[i]
}

func (a argsort) Less(i, j int) bool {
	return a.s[i] < a.s[j]
}

type argsort struct {
	s    []statmodel.Dtype
	inds []int
}

func (ph *PHReg) sortByStratum() {

	time := ph.data[ph.timepos]
	nobs := len(time)

	if ph.stratapos == -1 {
		ph.stratumix = [][2]int{{0, nobs}}
		return
	}

	strata := ph.data[ph.stratapos]

	inds := make([]int, nobs)
	for i := range inds {
		inds[i] = i
	}
	a := argsort{s: strata, inds: inds}
	sort.Sort(a)

	tmp := make([]statmodel.Dtype, nobs)

	re := func(pos int) {
		if pos == -1 {
			return
		}
		x := ph.data[pos]
		for i, j := range inds {
			tmp[i] = x[j]
		}
		x, tmp = tmp, x
		ph.data[pos] = x
	}

	re(ph.timepos)
	re(ph.statuspos)
	re(ph.offsetpos)
	re(ph.weightpos)
	re(ph.entrypos)

	for _, k := range ph.xpos {
		re(k)
	}

	var i0 int
	for i := 0; i <= len(strata); i++ {
		if i == len(strata) || (i > 0 && strata[i-1] != strata[i]) {
			ph.stratumix = append(ph.stratumix, [2]int{i0, i})
			i0 = i
		}
	}
}

func (ph *PHReg) setupTimes() {

	ph.skipEarlyCensor = 0

	time := ph.data[ph.timepos]
	status := ph.data[ph.statuspos]
	nobs := len(time)

	// Track cases that are omitted since they are
	// censored before the first event in their stratum.
	ph.skip = make([]bool, nobs)

	// Get the sorted distinct times where events occur
	for _, ix := range ph.stratumix {

		var et []float64

		for i := ix[0]; i < ix[1]; i++ {
			if time[i] < 0 {
				msg := fmt.Sprintf("PHReg: times cannot be negative.\n")
				panic(msg)
			}
			if status[i] == 1 {
				et = append(et, float64(time[i]))
			} else if status[i] != 0 {
				msg := fmt.Sprintf("PHReg: status variable '%s' has values other than 0 and 1.\n", ph.varnames[ph.statuspos])
				panic(msg)
			}
		}

		if len(et) > 0 {
			sort.Float64s(et)

			// Deduplicate
			j := 0
			for i := 1; i < len(et); i++ {
				if et[i] != et[j] {
					j++
					et[j] = et[i]
				}
			}
			et = et[0 : j+1]
		}
		ph.etimes = append(ph.etimes, et)

		// Indices of cases that enter or exit the risk set,
		// or have an event at each time point.
		enter := make([][]int, len(et))
		exit := make([][]int, len(et))
		event := make([][]int, len(et))
		ph.enter = append(ph.enter, enter)
		ph.exit = append(ph.exit, exit)
		ph.event = append(ph.event, event)

		// No events in this stratum
		if len(et) == 0 {
			continue
		}

		// Risk set exit times
		for i := ix[0]; i < ix[1]; i++ {
			ii := sort.SearchFloat64s(et, float64(time[i]))
			if ii < len(et) {
				if ii == len(et) {
					// Censored after last event, never exits
				} else if et[ii] == float64(time[i]) {
					// Event or censored at an event time
					exit[ii] = append(exit[ii], i)
				} else if ii == 0 {
					// Censored before first event, never enters
					ph.skip[i] = true
					ph.skipEarlyCensor++
				} else {
					// Censored between event times
					exit[ii-1] = append(exit[ii-1], i)
				}
			}
		}

		// Event times
		for i := ix[0]; i < ix[1]; i++ {
			if status[i] == 0 || ph.skip[i] {
				continue
			}
			ii := sort.SearchFloat64s(et, float64(time[i]))
			event[ii] = append(event[ii], i)
		}

		// Risk set entry times
		if ph.entrypos == -1 {
			// Everyone enters at time 0
			for i := ix[0]; i < ix[1]; i++ {
				if !ph.skip[i] {
					enter[0] = append(enter[0], i)
				}
			}
		} else {
			entry := ph.data[ph.entrypos]
			for i := ix[0]; i < ix[1]; i++ {
				if ph.skip[i] {
					continue
				}
				t := entry[i]
				if t > time[i] {
					msg := "PHReg: Entry times may not occur after event or censoring times.\n"
					panic(msg)
				}
				if t < 0 {
					msg := "PHReg: Entry times may not be negative.\n"
					panic(msg)

				}
				ii := sort.SearchFloat64s(et, float64(t))
				if ii == len(et) {
					// Enter after last event, never enters
				} else {
					// Enter on or between event times
					enter[ii] = append(enter[ii], i)
				}
			}
		}
	}
}

func (ph *PHReg) putNslice(x []float64) {
	ph.nslices = append(ph.nslices, x)
}

func (ph *PHReg) getNslice() []float64 {

	if len(ph.nslices) == 0 {
		return make([]float64, ph.NumObs())
	}
	q := len(ph.nslices) - 1
	x := ph.nslices[q]
	zero(x)
	ph.nslices = ph.nslices[0:q]

	return x
}

func (ph *PHReg) setupCovs() {

	ph.sumx = ph.sumx[0:0]
	status := ph.data[ph.statuspos]

	var wgt []statmodel.Dtype
	if ph.weightpos != -1 {
		wgt = ph.data[ph.weightpos]
	}

	// Get the sum of covariates in each stratum,
	// including only covariates for cases with the event
	sumx := make([]float64, len(ph.xpos))

	for _, ix := range ph.stratumix {
		for j, k := range ph.xpos {
			x := ph.data[k]
			for i := ix[0]; i < ix[1]; i++ {
				if !ph.skip[i] && status[i] == 1 {
					if wgt == nil {
						sumx[j] += float64(x[i])
					} else {
						sumx[j] += float64(wgt[i] * x[i])
					}
				}
			}
		}
		ph.sumx = append(ph.sumx, sumx)
		sumx = make([]float64, len(ph.xpos))
	}
}

// LogLike returns the log-likelihood at the given parameter value. The 'exact'
// parameter is ignored here.
func (ph *PHReg) LogLike(param statmodel.Parameter, exact bool) float64 {

	coeff := param.GetCoeff()

	ll := ph.breslowLogLike(coeff)

	// Account for L2 weights if present.
	if len(ph.l2wgt) > 0 {
		for j, x := range coeff {
			ll -= ph.l2wgt[j] * x * x
		}
	}

	return ll
}

// breslowLogLike returns the log-likelihood value for the
// proportional hazards regression model at the given parameter
// values, using the Breslow method to resolve ties.
func (ph *PHReg) breslowLogLike(params []float64) float64 {

	var wgt []statmodel.Dtype
	if ph.weightpos != -1 {
		wgt = ph.data[ph.weightpos]
	}

	var off []statmodel.Dtype
	if ph.offsetpos != -1 {
		off = ph.data[ph.offsetpos]
	}

	lp := ph.getNslice()
	elp := ph.getNslice()

	// Get the linear predictors
	for j, k := range ph.xpos {
		x := ph.data[k]
		for i := range x {
			lp[i] += float64(x[i]) * params[j]
		}
	}

	// Add the offset, if present
	if off != nil {
		for i := range off {
			lp[i] += float64(off[i])
		}
	}

	ql := float64(0)
	for s, ix := range ph.stratumix {

		// We can add any constant here due to invariance in
		// the partial likelihood.
		mx := floats.Max(lp[ix[0]:ix[1]])
		for i := ix[0]; i < ix[1]; i++ {
			lp[i] -= mx
			elp[i] = math.Exp(lp[i])
		}
		if wgt != nil {
			for i := ix[0]; i < ix[1]; i++ {
				lp[i] *= float64(wgt[i])
				elp[i] *= float64(wgt[i])
			}
		}

		rlp := float64(0)
		for k := 0; k < len(ph.etimes[s]); k++ {

			// Update for new entries
			for _, i := range ph.enter[s][k] {
				rlp += elp[i]
			}

			for _, i := range ph.event[s][k] {
				ql += lp[i]
			}

			if wgt != nil {
				var n float64
				for _, i := range ph.event[s][k] {
					n += float64(wgt[i])
				}
				ql -= n * math.Log(rlp)
			} else {
				ql -= float64(len(ph.event[s][k])) * math.Log(rlp)
			}

			// Update for new exits
			for _, i := range ph.exit[s][k] {
				rlp -= elp[i]
			}
		}
	}

	ph.putNslice(lp)
	ph.putNslice(elp)

	return ql
}

// BaselineCumHaz returns the Nelson-Aalen estimator of the baseline cumulative
// hazard function for the given stratum.
func (ph *PHReg) BaselineCumHaz(stratum int, params []float64) ([]float64, []float64) {

	h0 := make([]float64, len(ph.event[stratum]))

	ix := ph.stratumix[stratum]

	lp := make([]float64, ix[1]-ix[0])

	// Get the linear predictors for this stratum
	for j, k := range ph.xpos {
		x := ph.data[k]
		ii := 0
		for i := ix[0]; i < ix[1]; i++ {
			lp[ii] += float64(x[i]) * params[j]
			ii++
		}
	}

	elp := 0.0
	for k := range ph.etimes[stratum] {

		// Update for new entries
		for _, i := range ph.enter[stratum][k] {
			elp += math.Exp(lp[i-ix[0]])
		}

		h0[k] = float64(len(ph.event[stratum][k])) / elp

		// Update for new exits
		for _, i := range ph.exit[stratum][k] {
			elp -= math.Exp(lp[i-ix[0]])
		}
	}

	h1 := make([]float64, len(h0))
	for i := 1; i < len(h0); i++ {
		h1[i] = h1[i-1] + h0[i-1]
	}

	return ph.etimes[stratum], h1
}

func zero(x []float64) {
	for i := range x {
		x[i] = 0
	}
}

// Score computes the score vector for the proportional hazards
// regression model at the given parameter setting.
func (ph *PHReg) Score(params statmodel.Parameter, score []float64) {

	coeff := params.GetCoeff()
	ph.breslowScore(coeff, score)

	// Account for L2 weights if present.
	if len(ph.l2wgt) > 0 {
		for j, x := range coeff {
			score[j] -= 2 * ph.l2wgt[j] * x
		}
	}
}

// breslowScore calculates the score vector for the proportional
// hazards regression model at the given parameter values, using the
// Breslow approach to resolving ties.
func (ph *PHReg) breslowScore(params, score []float64) {

	zero(score)

	var wgt []statmodel.Dtype
	if ph.weightpos != -1 {
		wgt = ph.data[ph.weightpos]
	}

	var off []statmodel.Dtype
	if ph.offsetpos != -1 {
		off = ph.data[ph.offsetpos]
	}

	lp := ph.getNslice()

	// Get the linear predictors
	for j, k := range ph.xpos {
		x := ph.data[k]
		for i := range x {
			lp[i] += float64(x[i]) * params[j]
		}
	}

	if off != nil {
		for i := range off {
			lp[i] += float64(off[i])
		}
	}

	for s, ix := range ph.stratumix {

		if ph.sumx[s] == nil {
			continue
		}

		for j := 0; j < len(ph.xpos); j++ {
			score[j] += ph.sumx[s][j]
		}

		// We can add any constant here due to invariance in
		// the partial likelihood.
		mx := floats.Max(lp[ix[0]:ix[1]])
		for i := ix[0]; i < ix[1]; i++ {
			lp[i] = math.Exp(lp[i] - mx)
		}
		if wgt != nil {
			for i := ix[0]; i < ix[1]; i++ {
				lp[i] *= float64(wgt[i])
			}
		}

		rlp := float64(0)
		rlpv := make([]float64, len(ph.xpos))
		for q := range ph.etimes[s] {

			// Update for new entries
			for _, i := range ph.enter[s][q] {
				rlp += lp[i]
				for j, k := range ph.xpos {
					rlpv[j] += lp[i] * float64(ph.data[k][i])
				}
			}

			d := float64(len(ph.event[s][q]))
			if wgt != nil {
				d = 0
				for _, i := range ph.event[s][q] {
					d += float64(wgt[i])
				}
			}
			floats.AddScaledTo(score, score, -d/rlp, rlpv)

			// Update for new exits
			for _, i := range ph.exit[s][q] {
				rlp -= lp[i]
				for j, k := range ph.xpos {
					rlpv[j] -= lp[i] * float64(ph.data[k][i])
				}
			}
		}
	}

	ph.putNslice(lp)
}

// Hessian computes the Hessian matrix for the model evaluated at the
// given parameter setting.  The Hessian type parameter is not used
// here.
func (ph *PHReg) Hessian(params statmodel.Parameter, ht statmodel.HessType, hess []float64) {

	coeff := params.GetCoeff()
	ph.breslowHess(coeff, hess)

	// Account for L2 weights if present.
	p := len(coeff)
	if len(ph.l2wgt) > 0 {
		for j := 0; j < len(coeff); j++ {
			k := j*p + j
			hess[k] -= 2 * ph.l2wgt[j]
		}
	}
}

// breslowHess calculates the Hessian matrix for the proportional
// hazards regression model at the given parameter values.
func (ph *PHReg) breslowHess(params []float64, hess []float64) {

	zero(hess)

	var wgt []statmodel.Dtype
	if ph.weightpos != -1 {
		wgt = ph.data[ph.weightpos]
	}

	var off []statmodel.Dtype
	if ph.offsetpos != -1 {
		off = ph.data[ph.offsetpos]
	}

	time := ph.data[ph.timepos]
	nobs := len(time)
	lp := make([]float64, nobs)

	// Get the linear predictors
	for j, k := range ph.xpos {
		x := ph.data[k]
		for i := range x {
			lp[i] += float64(x[i]) * params[j]
		}
	}

	// Add the offset, if present
	if off != nil {
		for i := range off {
			lp[i] += float64(off[i])
		}
	}

	p := len(ph.xpos)
	d1s := make([]float64, p)
	d2s := make([]float64, p*p)

	for s, ix := range ph.stratumix {

		// We can add any constant here due to invariance in
		// the partial likelihood.
		mx := floats.Max(lp[ix[0]:ix[1]])
		for i := ix[0]; i < ix[1]; i++ {
			lp[i] = math.Exp(lp[i] - mx)
		}
		if wgt != nil {
			for i := ix[0]; i < ix[1]; i++ {
				lp[i] *= float64(wgt[i])
			}
		}

		rlp := float64(0)

		zero(d1s)
		zero(d2s)

		for k := 0; k < len(ph.etimes[s]); k++ {

			// Update for new entries
			for _, i := range ph.enter[s][k] {

				rlp += lp[i]

				for j1, k1 := range ph.xpos {
					x1 := ph.data[k1]
					d1s[j1] += lp[i] * float64(x1[i])
					for j2 := 0; j2 <= j1; j2++ {
						k2 := ph.xpos[j2]
						x2 := ph.data[k2]
						u := lp[i] * float64(x1[i]*x2[i])
						d2s[j1*p+j2] += u
						if j2 != j1 {
							d2s[j2*p+j1] += u
						}
					}
				}
			}

			d := float64(len(ph.event[s][k]))
			if wgt != nil {
				d = 0
				for _, i := range ph.event[s][k] {
					d += float64(wgt[i])
				}
			}

			jj := 0
			for j1 := 0; j1 < p; j1++ {
				for j2 := 0; j2 < p; j2++ {
					hess[jj] -= d * d2s[j1*p+j2] / rlp
					hess[jj] += d * d1s[j1] * d1s[j2] / (rlp * rlp)
					jj++
				}
			}

			// Update for new exits
			for _, i := range ph.exit[s][k] {

				rlp -= lp[i]
				for j1, k1 := range ph.xpos {
					x1 := ph.data[k1]
					d1s[j1] -= lp[i] * float64(x1[i])
					for j2 := 0; j2 <= j1; j2++ {
						k2 := ph.xpos[j2]
						x2 := ph.data[k2]
						u := lp[i] * float64(x1[i]*x2[i])
						d2s[j1*p+j2] -= u
						if j2 != j1 {
							d2s[j2*p+j1] -= u
						}
					}
				}
			}
		}
	}
}

func negative(x []float64) {
	for i := 0; i < len(x); i++ {
		x[i] *= -1
	}
}

// PHResults describes the results of a proportional hazards model..
type PHResults struct {
	statmodel.BaseResults
}

// failMessage prints information that can help diagnose optimization failures.
func (ph *PHReg) failMessage(optrslt *optimize.Result) {

	os.Stderr.WriteString("Current point and gradient:\n")
	for j, x := range optrslt.X {
		na := ph.varnames[ph.xpos[j]]
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", x, optrslt.Gradient[j], na))
	}

	time := ph.data[ph.timepos]
	status := ph.data[ph.statuspos]

	var entry []statmodel.Dtype
	if ph.entrypos != -1 {
		entry = ph.data[ph.entrypos]
	}

	var nEvent []float64
	var mTime []float64
	var stSize []float64
	var mEntry []float64

	mn := make([]float64, len(ph.xpos))
	sd := make([]float64, len(ph.xpos))

	for _, ix := range ph.stratumix {

		// Count the events per stratum
		var e, em float64
		for i := ix[0]; i < ix[1]; i++ {
			e += float64(status[i])
			em += float64(time[i])
		}
		nEvent = append(nEvent, e)
		mTime = append(mTime, em/float64(ix[1]-ix[0]))

		// Track the stratum sizes
		stSize = append(stSize, float64(ix[1]-ix[0]))

		// Get the mean entry time per stratum if available.
		if entry != nil {
			e := 0.0
			for i := ix[0]; i < ix[1]; i++ {
				e += float64(entry[i])
			}
			mEntry = append(mEntry, e/float64(ix[1]-ix[0]))
		}

		// Get the mean and standard deviation of covariates.
		for j, k := range ph.xpos {
			x := ph.data[k]
			for i := ix[0]; i < ix[1]; i++ {
				mn[j] += float64(x[i])
			}
			mn[j] /= float64(ix[1] - ix[0])
		}

		for j, k := range ph.xpos {
			x := ph.data[k]
			for i := ix[0]; i < ix[1]; i++ {
				u := float64(x[i]) - mn[j]
				sd[j] += u * u
			}
			sd[j] /= float64(ix[1] - ix[0])
			sd[j] = math.Sqrt(sd[j])
		}
	}

	os.Stderr.WriteString("\nCovariate means and standard deviations:\n")
	for j, m := range mn {
		na := ph.varnames[ph.xpos[j]]
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", m, sd[j], na))
	}

	os.Stderr.WriteString("\nStratum    Size       Events   Event_rate    Mean_time")
	if len(mEntry) > 0 {
		os.Stderr.WriteString("  Mean_entry\n")
	} else {
		os.Stderr.WriteString("\n")
	}
	for i, n := range stSize {
		os.Stderr.WriteString(fmt.Sprintf("%4d      %4.0f   %10.0f %12.3f %12.3f", i+1, n, nEvent[i], nEvent[i]/n, mTime[i]))
		if len(mEntry) > 0 {
			os.Stderr.WriteString(fmt.Sprintf(" %12.3f\n", mEntry[i]))
		} else {
			os.Stderr.WriteString("\n")
		}
	}
}

// Fit fits the model to the data.
func (ph *PHReg) Fit() (*PHResults, error) {

	if ph.l1wgt != nil {
		return ph.fitRegularized(), nil
	}

	nvar := len(ph.xpos)

	if ph.start == nil {
		ph.start = make([]float64, nvar)
	}

	p := optimize.Problem{
		Func: func(x []float64) float64 {
			return -ph.LogLike(&PHParameter{x}, false)
		},
		Grad: func(grad, x []float64) {
			if len(grad) != len(x) {
				grad = make([]float64, len(x))
			}
			ph.Score(&PHParameter{x}, grad)
			negative(grad)
		},
	}

	if ph.optsettings == nil {
		ph.optsettings = &optimize.Settings{
			GradientThreshold: 1e-5,
		}
	}

	var xna []string
	for _, k := range ph.xpos {
		xna = append(xna, ph.varnames[k])
	}

	optrslt, err := optimize.Minimize(p, ph.start, ph.optsettings, ph.optmethod)
	if err != nil {
		if optrslt == nil {
			return nil, err
		}

		// Return a partial results with an error
		results := &PHResults{
			BaseResults: statmodel.NewBaseResults(ph, -optrslt.F, optrslt.X, xna, nil),
		}
		ph.failMessage(optrslt)
		return results, err
	}
	if err = optrslt.Status.Err(); err != nil {
		return nil, err
	}

	param := make([]float64, len(optrslt.X))
	copy(param, optrslt.X)

	ll := -optrslt.F
	vcov, _ := statmodel.GetVcov(ph, &PHParameter{param})

	results := &PHResults{
		BaseResults: statmodel.NewBaseResults(ph, ll, param, xna, vcov),
	}

	return results, nil
}

// Focus returns a new PHReg instance with a single variable, which is variable j in the
// original model.  The effects of the remaining covariates are captured
// through the offset.
func (ph *PHReg) Focus(pos int, coeff []float64, offset []float64) statmodel.RegFitter {

	fph := *ph

	fph.varnames = []string{
		ph.varnames[ph.timepos],
		ph.varnames[ph.statuspos],
		ph.varnames[ph.xpos[pos]],
	}

	fph.data = [][]statmodel.Dtype{
		ph.data[ph.timepos],
		ph.data[ph.statuspos],
		ph.data[ph.xpos[pos]],
	}

	fph.timepos = 0
	fph.statuspos = 1
	fph.xpos = []int{2}
	fph.start = []float64{coeff[pos]}
	fph.log = ph.log

	// These are not used for coordinate optimization
	fph.optsettings = nil
	fph.optmethod = nil

	add := func(pos int) int {
		if pos == -1 {
			return -1
		}
		fph.varnames = append(fph.varnames, ph.varnames[pos])
		fph.data = append(fph.data, ph.data[pos])
		return len(fph.data) - 1
	}

	fph.weightpos = add(ph.weightpos)
	fph.entrypos = add(ph.entrypos)
	fph.stratapos = add(ph.stratapos)

	// Allocate a new slice for the offset
	nobs := ph.NumObs()
	if cap(offset) < nobs {
		offset = make([]float64, nobs)
	} else {
		offset = offset[0:nobs]
		zero(offset)
	}
	fph.varnames = append(ph.varnames, "__offset")
	fph.data = append(fph.data, make([]statmodel.Dtype, ph.NumObs()))
	fph.offsetpos = len(fph.data) - 1

	// Fill in the offset
	off := fph.data[fph.offsetpos]
	zerodtype(off)
	for j, k := range ph.xpos {
		if j != pos {
			for i := range off {
				off[i] += statmodel.Dtype(coeff[j] * float64(ph.data[k][i]))
			}
		}
	}

	// Add the original offset if present
	if ph.offsetpos != -1 {
		offsetOrig := ph.data[ph.offsetpos]
		for i := range offsetOrig {
			off[i] += offsetOrig[i]
		}
	}

	if ph.l2wgtMap != nil {
		fph.l2wgtMap = make(map[string]float64)
		vn := ph.varnames[ph.xpos[pos]]
		fph.l2wgtMap[vn] = ph.l2wgtMap[vn]
		fph.l2wgt = []float64{ph.l2wgtMap[vn]}
	} else {
		fph.l2wgt = nil
	}

	fph.l1wgtMap = nil
	fph.l1wgt = nil

	return &fph
}

func (rslt *PHResults) summaryStats() (int, int, int, int) {

	ph := rslt.Model().(*PHReg)
	data := ph.Dataset()

	status := data[ph.statuspos]

	var entry []statmodel.Dtype
	if ph.entrypos != -1 {
		entry = data[ph.entrypos]
	}

	var n, e, pe, ns int
	for _, ix := range ph.stratumix {
		n += ix[1] - ix[0]
		for i := ix[0]; i < ix[1]; i++ {
			e += int(status[i])
		}
		if entry != nil {
			for i := ix[0]; i < ix[1]; i++ {
				if entry[i] > 0 {
					pe++
				}
			}
		}
		ns++
	}

	return n, e, pe, ns
}

// fitRegularized estimates the parameters of the model using L1
// regularization (with optimal L2 regularization).  This invokes
// coordinate descent optimization.
func (ph *PHReg) fitRegularized() *PHResults {

	start := &PHParameter{
		coeff: make([]float64, len(ph.xpos)),
	}

	par := statmodel.FitL1Reg(ph, start, ph.l1wgt, ph.l2wgt, true)
	coeff := par.GetCoeff()

	// Covariate names
	var xna []string
	for _, j := range ph.xpos {
		xna = append(xna, ph.varnames[j])
	}

	results := &PHResults{
		BaseResults: statmodel.NewBaseResults(ph, 0, coeff, xna, nil),
	}

	return results
}

// PHSummary summarizes a fitted proportional hazards regression model.
type PHSummary struct {

	// The model
	ph *PHReg

	// The results structure
	results *PHResults

	// Transform the parameters with this function.  If nil,
	// no transformation is applied.  If paramXform is provided,
	// the standard error and Z-score are not shown.
	paramXform func(float64) float64

	// Messages that are appended to the table
	messages []string
}

// Summary displays a summary table of the model results.
func (rslt *PHResults) Summary() *PHSummary {

	ph := rslt.Model().(*PHReg)

	return &PHSummary{
		ph:      ph,
		results: rslt,
	}
}

// String returns a string representation of a summary table for the model.
func (phs *PHSummary) String() string {

	n, e, pe, ns := phs.results.summaryStats()

	ph := phs.ph
	sum := &statmodel.SummaryTable{
		Msg: phs.messages,
	}

	sum.Title = "Proportional hazards regression analysis"

	sum.Top = append(sum.Top, fmt.Sprintf("  Sample size: %10d", n))
	sum.Top = append(sum.Top, fmt.Sprintf("  Strata:      %10d", ns))
	sum.Top = append(sum.Top, fmt.Sprintf("  Events:      %10d", e))
	sum.Top = append(sum.Top, "  Ties:           Breslow")

	l1 := ph.l1wgt != nil

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

	fn := func(x interface{}, h string) []string {
		y := x.([]float64)
		var s []string
		for i := range y {
			s = append(s, fmt.Sprintf("%10.4f", y[i]))
		}
		return s
	}

	var hr []float64
	for j := range phs.results.Params() {
		hr = append(hr, math.Exp(phs.results.Params()[j]))
	}

	if !l1 && (phs.results.StdErr() != nil) {
		sum.ColNames = []string{"Variable   ", "Coefficient", "SE", "HR", "LCB", "UCB", "Z-score", "P-value"}
		sum.ColFmt = []statmodel.Fmter{fs, fn, fn, fn, fn, fn, fn, fn}

		// Create estimate and CI for the hazard ratio
		var lcb, ucb []float64
		for j := range phs.results.Params() {
			lcb = append(lcb, math.Exp(phs.results.Params()[j]-2*phs.results.StdErr()[j]))
			ucb = append(ucb, math.Exp(phs.results.Params()[j]+2*phs.results.StdErr()[j]))
		}
		sum.Cols = []interface{}{phs.results.Names(), phs.results.Params(), phs.results.StdErr(), hr, lcb, ucb,
			phs.results.ZScores(), phs.results.PValues()}
	} else {
		sum.ColNames = []string{"Variable   ", "Coefficient", "HR"}
		sum.ColFmt = []statmodel.Fmter{fs, fn, fn}
		sum.Cols = []interface{}{phs.results.Names(), phs.results.Params(), hr}
	}

	if pe > 0 {
		msg := fmt.Sprintf("%d observations have positive entry times", pe)
		sum.Msg = append(sum.Msg, msg)
	}

	if ph.skipEarlyCensor > 0 {
		msg := fmt.Sprintf("%d observations dropped for being censored before the first event\n", ph.skipEarlyCensor)
		sum.Msg = append(sum.Msg, msg)
	}

	return sum.String()
}
