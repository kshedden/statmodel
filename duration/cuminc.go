package duration

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/kshedden/dstream/dstream"
)

// CumincRight estimates the cumulative incidence functions for
// duration data with competing risks.
type CumincRight struct {

	// The data used to perform the estimation.
	data dstream.Dstream

	// The name of the variable containing the minimum of the
	// event time and entry time.  The underlying data must have
	// float64 type.
	timeVar string

	// The name of a variable containing the status indicator,
	// which is 1, 2, ... for the event types, and 0 for a
	// censored outcome.
	statusVar string

	// The name of a variable containing case weights, optional.
	weightVar string

	// The name of a variable containing entry times, optional.
	entryVar string

	// Times at which events occur, sorted.
	Times []float64

	// The number of occurrences of events of each type at each
	// time in Times.
	Events [][]float64

	// Number of events of any type at each time in Times
	EventsAll []float64

	// Risk set size at each time in times
	NRisk []float64

	// The estimated all-cause survival function
	ProbsAll []float64

	// The cause specific cumulative incidence rates.  Probs[k]
	// contains the rates for the events with Status==k+1
	// (Status==0 indicates censoring, cumulative incidences are
	// not estimated for the censored subjects).
	Probs [][]float64

	// The standard errors of the values in Probs
	ProbsSE [][]float64

	events    []map[float64]float64
	eventsall map[float64]float64
	total     map[float64]float64
	entry     map[float64]float64

	timePos   int
	statusPos int
	weightPos int
	entryPos  int

	varPos map[string]int
}

// NewCumincRight creates a CumincRight value that can be used to estimate
// the cumulative incidence function from the given data.
func NewCumincRight(data dstream.Dstream, timevar, statusvar string) *CumincRight {

	m := make(map[string]int)
	names := data.Names()
	for j, x := range names {
		m[x] = j
	}

	timepos, ok := m[timevar]
	if !ok {
		msg := fmt.Sprintf("Time variable '%s' not found.", timevar)
		panic(msg)
	}

	statuspos, ok := m[statusvar]
	if !ok {
		msg := fmt.Sprintf("Status variable '%s' not found.", statusvar)
		panic(msg)
	}

	return &CumincRight{
		data:      data,
		timeVar:   timevar,
		statusVar: statusvar,
		timePos:   timepos,
		statusPos: statuspos,
		weightPos: -1,
		entryPos:  -1,
		varPos:    m,
	}
}

// Weights specifies a variable that povides case weights.
func (ci *CumincRight) Weights(weightvar string) *CumincRight {

	var ok bool
	ci.weightPos, ok = ci.varPos[weightvar]
	if !ok {
		msg := fmt.Sprintf("Cannot find weight variable '%s'\n", weightvar)
		panic(msg)
	}
	ci.weightVar = weightvar

	return ci
}

// Entry specifies a variable that provides entry times.
func (ci *CumincRight) Entry(entryvar string) *CumincRight {

	var ok bool
	ci.entryPos, ok = ci.varPos[entryvar]
	if !ok {
		msg := fmt.Sprintf("Cannot find entry variable '%s'\n", entryvar)
		panic(msg)
	}
	ci.entryVar = entryvar

	return ci
}

func (ci *CumincRight) init() {
	ci.eventsall = make(map[float64]float64)
	ci.total = make(map[float64]float64)
	ci.entry = make(map[float64]float64)
	ci.data.Reset()
}

func (ci *CumincRight) scanData() {

	for j := 0; ci.data.Next(); j++ {

		time := ci.data.GetPos(ci.timePos).([]float64)
		status := ci.data.GetPos(ci.statusPos).([]float64)

		var entry []float64
		if ci.entryPos != -1 {
			entry = ci.data.GetPos(ci.entryPos).([]float64)
		}

		var weight []float64
		if ci.weightPos != -1 {
			weight = ci.data.GetPos(ci.weightPos).([]float64)
		}

		for i, t := range time {

			w := float64(1)
			if ci.weightPos != -1 {
				w = weight[i]
			}

			// Make room for an event type we have not yet seen
			k := int(status[i])
			for k > len(ci.events) {
				ci.events = append(ci.events, make(map[float64]float64))
			}

			if k > 0 {
				ci.events[k-1][t] += w
				ci.eventsall[t] += w
			}
			ci.total[t] += w

			if ci.entryPos != -1 {
				if entry[i] >= t {
					msg := fmt.Sprintf("Entry time %d in chunk %d is before the event/censoring times\n",
						i, j)
					os.Stderr.WriteString(msg)
					os.Exit(1)
				}
				ci.entry[entry[i]] += w
			}
		}
	}
}

func (ci *CumincRight) eventstats() {

	// Get the sorted times (event or censoring)
	ci.Times = make([]float64, len(ci.total))
	var i int
	for t := range ci.total {
		ci.Times[i] = t
		i++
	}
	sort.Float64s(ci.Times)

	// Get the weighted event count and risk set size at each time
	// point (in same order as Times).
	ci.EventsAll = make([]float64, len(ci.Times))
	ci.NRisk = make([]float64, len(ci.Times))
	for i, t := range ci.Times {
		ci.EventsAll[i] = ci.eventsall[t]
		ci.NRisk[i] = ci.total[t]
	}
	rollback(ci.NRisk)

	// Adjust for entry times
	if ci.entryPos != -1 {
		entry := make([]float64, len(ci.Times))
		for t, w := range ci.entry {
			ii := sort.SearchFloat64s(ci.Times, t)
			if t < ci.Times[ii] {
				ii--
			}
			if ii >= 0 {
				entry[ii] += w
			}
		}
		rollback(entry)
		for i := 0; i < len(ci.NRisk); i++ {
			ci.NRisk[i] -= entry[i]
		}
	}
}

func (ci *CumincRight) fitall() {

	ci.ProbsAll = make([]float64, len(ci.Times))

	x := float64(1)
	for i := range ci.Times {
		x *= 1 - ci.EventsAll[i]/ci.NRisk[i]
		ci.ProbsAll[i] = x
	}
}

func (ci *CumincRight) fit() {

	for _, ev := range ci.events {

		// Obtain the number of events of each cause at each time.
		evr := make([]float64, len(ci.Times))
		for t, n := range ev {
			ii := sort.SearchFloat64s(ci.Times, t)
			evr[ii] += n
		}

		cir := make([]float64, len(ci.Times))
		x := float64(0)
		for i, y := range evr {
			v := y / ci.NRisk[i]
			if i > 0 {
				v *= ci.ProbsAll[i-1]
			}
			x += v
			cir[i] = x
		}

		ci.Probs = append(ci.Probs, cir)
		ci.Events = append(ci.Events, evr)
	}
}

// DEBUG fix failing TestCI2
func (ci *CumincRight) fitse() {

	ngrp := len(ci.Probs)

	for k := 0; k < ngrp; k++ {

		var x1, x2, x3, x4, x5, x6 float64
		se := make([]float64, len(ci.Times))

		for i := range ci.Times {

			q := ci.Probs[k][i]
			da := ci.EventsAll[i]
			d := ci.Events[k][i]
			n := ci.NRisk[i]
			s := float64(1)
			if i > 0 {
				s = ci.ProbsAll[i-1]
			}
			s /= n

			ra := da / (n * (n - da))
			x1 += ra
			x2 += q * ra
			x3 += q * q * ra

			ra = (n - d) * d / n
			x4 += s * s * ra

			ra = s * d / n
			x5 += ra
			x6 += q * ra

			v := q*q*x1 - 2*q*x2 + x3 + x4 - 2*q*x5 + 2*x6
			se[i] = math.Sqrt(v)
		}

		ci.ProbsSE = append(ci.ProbsSE, se)
	}
}

// compress removes times where no events occurred.
func (ci *CumincRight) compress() {

	var ix []int
	for i := 0; i < len(ci.Times); i++ {
		if ci.EventsAll[i] > 0 {
			ix = append(ix, i)
		}
	}

	if len(ix) < len(ci.Times) {
		for i, j := range ix {
			ci.Times[i] = ci.Times[j]
			ci.EventsAll[i] = ci.EventsAll[j]
			ci.NRisk[i] = ci.NRisk[j]
		}
		ci.Times = ci.Times[0:len(ix)]
		ci.EventsAll = ci.EventsAll[0:len(ix)]
		ci.NRisk = ci.NRisk[0:len(ix)]
	}
}

// Done completes construction and computes all results.
func (ci *CumincRight) Done() *CumincRight {
	ci.init()
	ci.scanData()
	ci.eventstats()
	ci.compress()
	ci.fitall()
	ci.fit()
	ci.fitse()
	return ci
}
