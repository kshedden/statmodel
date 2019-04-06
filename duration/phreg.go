// Package duration supports various methods for statistical analysis
// of duration data (survival analysis).
package duration

import (
	"fmt"
	"math"
	"os"
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/optimize"

	"github.com/kshedden/dstream/dstream"
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

	// The data to which the model is fit
	data dstream.Dstream

	// Starting values, optional
	start []float64

	// Name and position of the event variable
	statusvar    string
	statusvarpos int

	// Name and position of the time variable
	timevar    string
	timevarpos int

	// Name and position of the entry time variable
	entryvar    string
	entryvarpos int

	// Name and position of an offset variable
	offsetvar    string
	offsetvarpos int

	// Name and positin of a case weight variable
	weightvarpos int
	weightvar    string

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

	// The L2 norms of the centered covariates.  If norm=true,
	// calculations are done on normalized covariates.
	xscale []float64

	// The mean of every covariate, used to center the covariates.
	// Since there is no intercept in a Cox model, the covariates
	// can be centered without further adjustment.
	xmean []float64

	// The approach for scaling the covariates
	scaletype statmodel.ScaleType

	// The sum of covariates with events in each stratum
	sumx [][]float64

	// L2 (ridge) weights for each variable
	l2wgt []float64

	// L1 (lasso) weights for each variable
	l1wgt []float64

	// The positions of the covariates in the Dstream
	xpos []int

	// If skip[s][i] is true, case i within stratum j is skipped
	// since it is censored before the first event.
	skip []map[int]bool

	// Optimization settings
	settings *optimize.Settings

	// Optimization method
	method optimize.Method

	// Indicates that Done has already been called
	done bool
}

// The number of parameters (regression coefficients).
func (ph *PHReg) NumParams() int {
	return len(ph.xpos)
}

// Weight sets the case weights to be used when fitting the model.
func (ph *PHReg) Weight(name string) *PHReg {
	ph.weightvar = name
	return ph
}

// L2Weight sets L2 (ridge) weights to be used when fitting the model.
// When using regularization, consider calling ScaleType so that the
// penalties have the same impact on all covariates.
func (ph *PHReg) L2Weight(w []float64) *PHReg {
	ph.l2wgt = w
	return ph
}

// L1Weight sets L1 (lasso) weights to be used when fitting the model.
// When using regularization, consider calling ScaleType so that the
// penalties have the same impact on all covariates.
func (ph *PHReg) L1Weight(w []float64) *PHReg {
	ph.l1wgt = w
	return ph
}

// CovariateScale sets the scaling method for the covariates.
func (ph *PHReg) CovariateScale(scaletype statmodel.ScaleType) *PHReg {
	ph.scaletype = scaletype
	return ph
}

// The name of an offset variable
func (ph *PHReg) Offset(off string) *PHReg {
	ph.offsetvar = off
	return ph
}

// The positions of the covariates in the model's data..
func (ph *PHReg) Xpos() []int {
	return ph.xpos
}

// OptSettings allows the caller to provide an optimization settings
// value.
func (ph *PHReg) OptSettings(s *optimize.Settings) *PHReg {
	ph.settings = s
	return ph
}

// NewPHReg returns a PHReg value that can be used to fit a
// proportional hazards regression model.  Call configuration methods
// (Entry, etc.) then call Done before fitting.
func NewPHReg(data dstream.Dstream, time, status string) *PHReg {
	ph := &PHReg{
		data:      data,
		timevar:   time,
		statusvar: status,
	}

	return ph
}

// Entry sets the delayed entry times to be used when fitting the model.
func (ph *PHReg) Entry(entry string) *PHReg {
	ph.entryvar = entry
	return ph
}

// Start sets starting values to be used when fitting the model.
func (ph *PHReg) Start(start []float64) *PHReg {
	ph.start = start
	return ph
}

// Done signals that configuration is complete and the model can be fit.
func (ph *PHReg) Done() *PHReg {
	ph.findvars()
	ph.doScale()
	ph.setupTimes()
	ph.setupCovs()

	if ph.scaletype != statmodel.NoScale && ph.start != nil {
		for j := range ph.start {
			ph.start[j] *= ph.xscale[j]
		}
	}

	ph.done = true
	return ph
}

func (ph *PHReg) findvars() {
	ph.statusvarpos, ph.timevarpos, ph.entryvarpos, ph.offsetvarpos, ph.weightvarpos = -1, -1, -1, -1, -1
	for i, na := range ph.data.Names() {
		switch na {
		case ph.statusvar:
			ph.statusvarpos = i
		case ph.timevar:
			ph.timevarpos = i
		case ph.entryvar:
			ph.entryvarpos = i
		case ph.offsetvar:
			ph.offsetvarpos = i
		case ph.weightvar:
			ph.weightvarpos = i
		default:
			// Everything else is a covariate
			ph.xpos = append(ph.xpos, i)
		}
	}

	if ph.timevarpos == -1 {
		msg := fmt.Sprintf("Time variable '%s' not found\n", ph.timevar)
		panic(msg)
	}
	if ph.statusvarpos == -1 {
		msg := fmt.Sprintf("Event status variable '%s' not found\n", ph.statusvar)
		panic(msg)
	}
	if ph.entryvar != "" && ph.entryvarpos == -1 {
		msg := fmt.Sprintf("Entry variable '%s' not found\n", ph.entryvar)
		panic(msg)
	}
	if ph.offsetvar != "" && ph.offsetvarpos == -1 {
		msg := fmt.Sprintf("Offset variable '%s' not found\n", ph.offsetvar)
		panic(msg)
	}
	if ph.weightvar != "" && ph.weightvarpos == -1 {
		msg := fmt.Sprintf("Weight variable '%s' not found\n", ph.weightvar)
		panic(msg)
	}
}

func (ph *PHReg) doScale() {

	ph.xmean = make([]float64, len(ph.xpos))
	ph.xscale = make([]float64, len(ph.xpos))

	// Get the centers of the covariates
	ph.data.Reset()
	var n float64
	for ph.data.Next() {

		var wgt []float64
		if ph.weightvarpos != -1 {
			wgt = ph.data.GetPos(ph.weightvarpos).([]float64)
		}

		for j, k := range ph.xpos {
			x := ph.data.GetPos(k).([]float64)
			for i := range x {
				if wgt == nil {
					ph.xmean[j] += x[i]
					n++
				} else {
					ph.xmean[j] += wgt[i] * x[i]
					n += wgt[i]
				}
			}
		}
	}

	floats.Scale(1/n, ph.xmean)

	if ph.scaletype == statmodel.NoScale {
		for k := range ph.xscale {
			ph.xscale[k] = 1
		}
		return
	}

	// Calculate the L2 norms of the covariates.
	ph.data.Reset()
	for ph.data.Next() {

		var wgt []float64
		if ph.weightvarpos != -1 {
			wgt = ph.data.GetPos(ph.weightvarpos).([]float64)
		}

		for j, k := range ph.xpos {
			x := ph.data.GetPos(k).([]float64)
			for i := range x {
				u := x[i] - ph.xmean[j]
				if wgt != nil {
					ph.xscale[j] += wgt[i] * u * u
				} else {
					ph.xscale[j] += u * u
				}
			}
		}
	}

	for j := range ph.xscale {

		switch ph.scaletype {
		case statmodel.L2Norm:
			ph.xscale[j] = math.Sqrt(ph.xscale[j])
		case statmodel.Variance:
			ph.xscale[j] = math.Sqrt(ph.xscale[j] / n)
		default:
			panic("Unkown scale type")
		}

		if ph.xscale[j] == 0 {
			names := ph.data.Names()
			name := names[ph.xpos[j]]
			msg := fmt.Sprintf("Variable %s has zero variance.\n", name)
			panic(msg)
		}
	}
}

func (ph *PHReg) setupTimes() {

	ph.data.Reset()
	nskip := 0

	// Each chunk is a stratum
	for s := 0; ph.data.Next(); s++ {

		// Track cases that are omitted since they are
		// censored before the first event in their stratum.
		ph.skip = append(ph.skip, make(map[int]bool))

		// Get the time data
		tim := ph.data.GetPos(ph.timevarpos).([]float64)

		// Get the event status data
		status := ph.data.GetPos(ph.statusvarpos).([]float64)

		// The sorted distinct times where events occur
		var et []float64
		for i, t := range tim {
			if t < 0 {
				msg := fmt.Sprintf("PHReg: times cannot be negative.\n")
				panic(msg)
			}
			if status[i] == 1 {
				et = append(et, t)
			} else if status[i] != 0 {
				msg := fmt.Sprintf("PHReg: status variable '%s' has values other than 0 and 1.\n", ph.statusvar)
				panic(msg)
			}
		}
		if len(et) > 0 {
			sort.Sort(sort.Float64Slice(et))
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
		for i, t := range tim {
			ii := sort.SearchFloat64s(et, t)
			if ii < len(et) {
				if ii == len(et) {
					// Censored after last event, never exits
				} else if et[ii] == t {
					// Event or censored at an event time
					exit[ii] = append(exit[ii], i)
				} else if ii == 0 {
					// Censored before first event, never enters
					ph.skip[s][i] = true
					nskip++
				} else {
					// Censored between event times
					exit[ii-1] = append(exit[ii-1], i)
				}
			}
		}

		// Event times
		for i, t := range tim {
			if status[i] == 0 || ph.skip[s][i] {
				continue
			}
			ii := sort.SearchFloat64s(et, t)
			event[ii] = append(event[ii], i)
		}

		// Risk set entry times
		if ph.entryvarpos == -1 {
			// Everyone enters at time 0
			for i := range tim {
				if !ph.skip[s][i] {
					enter[0] = append(enter[0], i)
				}
			}
		} else {
			ent := ph.data.GetPos(ph.entryvarpos).([]float64)
			for i, t := range ent {
				if ph.skip[s][i] {
					continue
				}
				if t > tim[i] {
					msg := "PHReg: Entry times may not occur after event or censoring times.\n"
					panic(msg)
				}
				if t < 0 {
					msg := "PHReg: Entry times may not be negative.\n"
					panic(msg)

				}
				ii := sort.SearchFloat64s(et, t)
				if ii == len(et) {
					// Enter after last event, never enters
				} else {
					// Enter on or between event times
					enter[ii] = append(enter[ii], i)
				}
			}
		}
	}

	if nskip > 0 {
		fmt.Printf("%d observations dropped for being censored before the first event\n", nskip)
	}
}

func (ph *PHReg) setupCovs() {

	ph.sumx = ph.sumx[0:0]
	ph.data.Reset()
	for s := 0; ph.data.Next(); s++ {

		status := ph.data.GetPos(ph.statusvarpos).([]float64)

		var wgt []float64
		if ph.weightvarpos != -1 {
			wgt = ph.data.GetPos(ph.weightvarpos).([]float64)
		}

		// Get the sum of covariates in each stratum,
		// including only covariates for cases with the event
		sumx := make([]float64, len(ph.xpos))
		for k, j := range ph.xpos {
			x := ph.data.GetPos(j).([]float64)
			for i, v := range x {
				if !ph.skip[s][i] && status[i] == 1 {
					if wgt == nil {
						sumx[k] += (v - ph.xmean[k]) / ph.xscale[k]
					} else {
						sumx[k] += wgt[i] * (v - ph.xmean[k]) / ph.xscale[k]
					}
				}
			}
		}
		ph.sumx = append(ph.sumx, sumx)
	}
}

// LogLike returns the log-likelihood at the given parameter value.
func (ph *PHReg) LogLike(param statmodel.Parameter) float64 {

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

	ph.data.Reset()

	ql := float64(0)
	for s := 0; ph.data.Next(); s++ {

		tim := ph.data.GetPos(ph.timevarpos).([]float64)
		lp := make([]float64, len(tim))

		var wgt []float64
		if ph.weightvarpos != -1 {
			wgt = ph.data.GetPos(ph.weightvarpos).([]float64)
		}

		// Get the linear predictors
		for k, j := range ph.xpos {
			x := ph.data.GetPos(j).([]float64)
			for i, v := range x {
				lp[i] += (v - ph.xmean[k]) * params[k] / ph.xscale[k]
			}
		}

		// Add the offset, if present
		if ph.offsetvarpos != -1 {
			off := ph.data.GetPos(ph.offsetvarpos).([]float64)
			floats.AddTo(lp, lp, off)
		}

		// We can add any constant here due to invariance in
		// the partial likelihood.
		mx := floats.Max(lp)
		for i := range lp {
			lp[i] -= mx
		}

		rlp := float64(0)
		for k := 0; k < len(ph.etimes[s]); k++ {

			// Update for new entries
			for _, i := range ph.enter[s][k] {
				if wgt != nil {
					rlp += wgt[i] * math.Exp(lp[i])
				} else {
					rlp += math.Exp(lp[i])
				}
			}

			for _, i := range ph.event[s][k] {
				if wgt != nil {
					ql += wgt[i] * lp[i]
				} else {
					ql += lp[i]
				}
			}

			if wgt != nil {
				var n float64
				for _, i := range ph.event[s][k] {
					n += wgt[i]
				}
				ql -= n * math.Log(rlp)
			} else {
				ql -= float64(len(ph.event[s][k])) * math.Log(rlp)
			}

			// Update for new exits
			for _, i := range ph.exit[s][k] {
				if wgt != nil {
					rlp -= wgt[i] * math.Exp(lp[i])
				} else {
					rlp -= math.Exp(lp[i])
				}
			}
		}
	}

	return ql
}

// BaselineCumHaz returns the Nelson-Aalen estimator of the baseline cumulative
// hazard function for the given stratum.
func (ph *PHReg) BaselineCumHaz(stratum int, params []float64) ([]float64, []float64) {

	h0 := make([]float64, len(ph.event[stratum]))

	ph.data.Reset()
	var tim, lp []float64
	for s := 0; ph.data.Next(); s++ {

		// Use only the chosen stratum.
		if s != stratum {
			continue
		}

		tim = ph.data.GetPos(ph.timevarpos).([]float64)
		lp = make([]float64, len(tim))

		// Get the linear predictors
		for k, j := range ph.xpos {
			x := ph.data.GetPos(j).([]float64)
			for i, v := range x {
				lp[i] += (v - ph.xmean[k]) * params[k] / ph.xscale[k]
			}
		}
	}

	elp := 0.0
	for k := 0; k < len(ph.etimes[stratum]); k++ {

		// Update for new entries
		for _, i := range ph.enter[stratum][k] {
			elp += math.Exp(lp[i])
		}

		h0[k] = float64(len(ph.event[stratum][k])) / elp

		// Update for new exits
		for _, i := range ph.exit[stratum][k] {
			elp -= math.Exp(lp[i])
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

	ph.data.Reset()

	zero(score)

	for s := 0; ph.data.Next(); s++ {

		if ph.sumx[s] == nil {
			continue
		}

		var wgt []float64
		if ph.weightvarpos != -1 {
			wgt = ph.data.GetPos(ph.weightvarpos).([]float64)
		}

		// Get the covariates to avoid repeated type assertions
		var xvars [][]float64
		for _, j := range ph.xpos {
			xvars = append(xvars, ph.data.GetPos(j).([]float64))
		}

		for j := 0; j < len(ph.xpos); j++ {
			score[j] += ph.sumx[s][j]
		}

		tim := ph.data.GetPos(ph.timevarpos).([]float64)
		lp := make([]float64, len(tim))

		// Get the linear predictors
		for k := range ph.xpos {
			x := xvars[k]
			for i, v := range x {
				lp[i] += (v - ph.xmean[k]) * params[k] / ph.xscale[k]
			}
		}

		// Add the offset, if present
		if ph.offsetvarpos != -1 {
			off := ph.data.GetPos(ph.offsetvarpos).([]float64)
			floats.AddTo(lp, lp, off)
		}

		// We can add any constant here due to invariance in
		// the partial likelihood.
		mx := floats.Max(lp)
		for i := range lp {
			lp[i] -= mx
		}

		rlp := float64(0)
		rlpv := make([]float64, len(ph.xpos))
		for k := 0; k < len(ph.etimes[s]); k++ {

			// Update for new entries
			for _, i := range ph.enter[s][k] {
				f := math.Exp(lp[i])
				if wgt != nil {
					f *= wgt[i]
				}
				rlp += f
				for j, x := range xvars {
					rlpv[j] += f * (x[i] - ph.xmean[j]) / ph.xscale[j]
				}
			}

			d := float64(len(ph.event[s][k]))
			if wgt != nil {
				d = 0
				for _, i := range ph.event[s][k] {
					d += wgt[i]
				}
			}
			floats.AddScaledTo(score, score, -d/rlp, rlpv)

			// Update for new exits
			for _, i := range ph.exit[s][k] {
				f := math.Exp(lp[i])
				if wgt != nil {
					f *= wgt[i]
				}
				rlp -= f
				for j, x := range xvars {
					rlpv[j] -= f * (x[i] - ph.xmean[j]) / ph.xscale[j]
				}
			}
		}
	}
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

	ph.data.Reset()

	zero(hess)

	for s := 0; ph.data.Next(); s++ {

		var wgt []float64
		if ph.weightvarpos != -1 {
			wgt = ph.data.GetPos(ph.weightvarpos).([]float64)
		}

		// Get the covariates to avoid repeated type assertions
		var xvars [][]float64
		for _, j := range ph.xpos {
			xvars = append(xvars, ph.data.GetPos(j).([]float64))
		}

		tim := ph.data.GetPos(ph.timevarpos).([]float64)
		lp := make([]float64, len(tim))

		// Get the linear predictors
		for k := range ph.xpos {
			x := xvars[k]
			for i, v := range x {
				lp[i] += v * params[k]
			}
		}

		// Add the offset, if present
		if ph.offsetvarpos != -1 {
			off := ph.data.GetPos(ph.offsetvarpos).([]float64)
			floats.AddTo(lp, lp, off)
		}

		rlp := float64(0)
		p := len(xvars)
		d1s := make([]float64, p)
		d2s := make([]float64, p*p)
		for k := 0; k < len(ph.etimes[s]); k++ {

			// Update for new entries
			for _, i := range ph.enter[s][k] {

				f := math.Exp(lp[i])
				if wgt != nil {
					f *= wgt[i]
				}
				rlp += f

				for j1, x1 := range xvars {
					d1s[j1] += f * x1[i]
					for j2 := 0; j2 <= j1; j2++ {
						x2 := xvars[j2]
						u := f * x1[i] * x2[i]
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
					d += wgt[i]
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

				f := math.Exp(lp[i])
				if wgt != nil {
					f *= wgt[i]
				}

				rlp -= f
				for j1, x1 := range xvars {
					d1s[j1] -= f * x1[i]
					for j2 := 0; j2 <= j1; j2++ {
						x2 := xvars[j2]
						u := f * x1[i] * x2[i]
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

// DataSet returns the underlying data set for the regression model.
func (ph *PHReg) DataSet() dstream.Dstream {
	return ph.data
}

// failMessage prints information that can help diagnose optimization failures.
func (ph *PHReg) failMessage(optrslt *optimize.Result) {

	xnames := ph.data.Names()

	os.Stderr.WriteString("Current point and gradient:\n")
	for j, x := range optrslt.X {
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", x, optrslt.Gradient[j], xnames[ph.xpos[j]]))
	}

	// Get the covariates to avoid repeated type assertions
	ph.data.Reset()
	xvars := make([][]float64, len(ph.xpos))
	var nEvent []float64
	var mTime []float64
	var stSize []float64
	var mEntry []float64
	for ph.data.Next() {
		for k, j := range ph.xpos {
			xvars[k] = append(xvars[k], ph.data.GetPos(j).([]float64)...)
		}

		// Count the events per stratum
		status := ph.data.GetPos(ph.statusvarpos).([]float64)
		e := floats.Sum(status)
		nEvent = append(nEvent, e)

		// Get the mean event time per stratum
		time := ph.data.GetPos(ph.timevarpos).([]float64)
		e = floats.Sum(time) / float64(len(time))
		mTime = append(mTime, e)

		// Track the stratum sizes
		stSize = append(stSize, float64(len(time)))

		// Get the mean entry time per stratum if avaiable.
		if ph.entryvarpos != -1 {
			entry := ph.data.GetPos(ph.entryvarpos).([]float64)
			e = floats.Sum(entry) / float64(len(entry))
			mEntry = append(mEntry, e)
		}
	}

	// Get the mean and standard deviation of covariates.
	mn := make([]float64, len(ph.xpos))
	sd := make([]float64, len(ph.xpos))
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
		os.Stderr.WriteString(fmt.Sprintf("%16.8f %16.8f %s\n", m, sd[j], xnames[ph.xpos[j]]))
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

// Optimizer sets the optimization method from gonum.Optimize.
func (ph *PHReg) Optimizer(method optimize.Method) *PHReg {
	ph.method = method
	return ph
}

// Fit fits the model to the data.
func (ph *PHReg) Fit() (*PHResults, error) {

	if ph.l1wgt != nil {
		return ph.fitRegularized(), nil
	}

	if !ph.done {
		msg := fmt.Sprintf("PHReg: must call Done before calling Fit")
		panic(msg)
	}

	nvar := len(ph.xpos)

	if ph.start == nil {
		ph.start = make([]float64, nvar)
	}

	p := optimize.Problem{
		Func: func(x []float64) float64 {
			return -ph.LogLike(&PHParameter{x})
		},
		Grad: func(grad, x []float64) []float64 {
			if len(grad) != len(x) {
				grad = make([]float64, len(x))
			}
			ph.Score(&PHParameter{x}, grad)
			negative(grad)
			return grad
		},
	}

	if ph.settings == nil {
		ph.settings = &optimize.Settings{}
		ph.settings.Recorder = nil
		ph.settings.GradientThreshold = 1e-5
	}

	if ph.method == nil {
		ph.method = &optimize.BFGS{}
	}

	var xna []string
	na := ph.data.Names()
	for _, k := range ph.xpos {
		xna = append(xna, na[k])
	}

	optrslt, err := optimize.Minimize(p, ph.start, ph.settings, ph.method)
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
	for j := range optrslt.X {
		param[j] = optrslt.X[j] / ph.xscale[j]
	}

	ll := -optrslt.F
	vcov, _ := statmodel.GetVcov(ph, &PHParameter{param})

	results := &PHResults{
		BaseResults: statmodel.NewBaseResults(ph, ll, param, xna, vcov),
	}

	return results, nil
}

// Get a focusable version of the model, which can be projected onto
// the coordinate axes for coordinate optimization.  It is exposed for
// use in elastic net optimization, but unlikely to be useful to
// ordinary users.
func (ph *PHReg) GetFocusable() statmodel.ModelFocuser {

	other := []string{ph.timevar, ph.statusvar}

	// Set up the focusable data.
	fdat := statmodel.NewFocusData(ph.data, ph.xpos, ph.xscale).Other(other).Done()

	newph := NewPHReg(fdat, ph.timevar, ph.statusvar).Offset("off")

	if ph.l1wgt != nil {
		newph = newph.L1Weight([]float64{0})
	}

	if ph.l2wgt != nil {
		newph = newph.L2Weight([]float64{0})
	}

	if ph.entryvar != "" {
		newph = newph.Entry(ph.entryvar)
	}

	newph = newph.Done()

	return newph
}

// Focus sets the data to contain only one predictor (with the given
// index).  The effects of the remaining covariates are captured
// throughthe offset.  The is exposed for use in elastic net fitting
// but is unlikely to be useful for ordinary users.  Can only be
// called on a focusable version of the model value.
func (ph *PHReg) Focus(j int, coeff []float64, l2wgt float64) {
	ph.data.(*statmodel.FocusData).Focus(j, coeff)

	if l2wgt > 0 {
		ph.l2wgt[0] = l2wgt
	}

	// Need to rerun these after adjusting covariates
	ph.doScale()
	ph.setupCovs()
}

func (rslt *PHResults) summaryStats() (int, int, int, int) {

	ph := rslt.Model().(*PHReg)
	data := ph.DataSet()

	var n, e, pe, ns int
	data.Reset()
	for data.Next() {
		st := data.GetPos(ph.statusvarpos).([]float64)
		n += len(st)
		for _, x := range st {
			e += int(x)
		}
		if ph.entryvarpos != -1 {
			et := data.GetPos(ph.entryvarpos).([]float64)
			for _, x := range et {
				if x > 0 {
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

	par := statmodel.FitL1Reg(ph, start, ph.l1wgt, ph.l2wgt, ph.xscale, true)
	coeff := par.GetCoeff()

	// Since coeff is transformed back to the original scale, we
	// need to stop normalizing now.
	for i, _ := range ph.xscale {
		ph.xscale[i] = 1
	}

	// Covariate names
	var xna []string
	na := ph.data.Names()
	for _, j := range ph.xpos {
		xna = append(xna, na[j])
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

	return sum.String()
}
