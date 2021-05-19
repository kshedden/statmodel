package duration

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/kshedden/statmodel/statmodel"
)

// SurvfuncRight uses the method of Kaplan and Meier to estimate the
// survival distribution based on (possibly) right censored data.  The
// caller must set Data and TimeVar before calling the Fit method.
// StatusVar, WeightVar, and EntryVar are optional fields.
type SurvfuncRight struct {
	data [][]float64

	// The name of the variable containing the minimum of the
	// event time and entry time.
	timepos int

	// The name of a variable containing the status indicator,
	// which is 1 if the event occurred at the time given by
	// TimeVar, and 0 otherwise.  This is optional, and is assumed
	// to be identically equal to 1 if not present.
	statuspos int

	// The name of a variable containing case weights, optional.
	weightpos int

	// The name of a variable containing entry times, optional.
	entrypos int

	// Times at which events occur, sorted.
	times []float64

	// Number of events at each time in Times.
	nEvents []float64

	// Number of people at risk just before each time in times
	nRisk []float64

	// The estimated survival function evaluated at each time in Times
	survProb []float64

	// The standard errors for the estimates in SurvProb.
	survProbSE []float64

	events map[float64]float64
	total  map[float64]float64
	entry  map[float64]float64
}

type SurvfuncRightConfig struct {
	WeightVar string
	EntryVar  string
}

// NewSurvfuncRight creates a new value for fitting a survival function.
func NewSurvfuncRight(data statmodel.Dataset, time, status string, config *SurvfuncRightConfig) (*SurvfuncRight, error) {

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

	getpos := func(cfg *SurvfuncRightConfig, field string) int {
		if cfg == nil {
			return -1
		}

		var vn string
		switch field {
		case "weight":
			vn = config.WeightVar
		case "entry":
			vn = config.EntryVar
		default:
			panic("!!")
		}

		if vn == "" {
			return -1
		}

		loc, ok := pos[vn]
		if !ok {
			msg := fmt.Sprintf("'%s' not found\n", vn)
			panic(msg)
		}

		return loc
	}

	return &SurvfuncRight{
		data:      data.Data(),
		timepos:   timepos,
		statuspos: statuspos,
		weightpos: getpos(config, "weight"),
		entrypos:  getpos(config, "entry"),
	}, nil
}

func (sf *SurvfuncRight) Fit() {

	sf.init()
	sf.scanData()
	sf.eventstats()
	sf.compress()
	sf.fit()
}

// NumRisk returns the number of people at risk at each time point
// where the survival function changes.
func (sf *SurvfuncRight) NumRisk() []float64 {
	return sf.nRisk
}

// SurvProb returns the estimated survival probabilities at the points
// where the survival function changes.
func (sf *SurvfuncRight) SurvProb() []float64 {
	return sf.survProb
}

// SurvProbSE returns the standard errors of the estimated survival
// probabilities at the points where the survival function changes.
func (sf *SurvfuncRight) SurvProbSE() []float64 {
	return sf.survProbSE
}

func (sf *SurvfuncRight) init() {

	sf.events = make(map[float64]float64)
	sf.total = make(map[float64]float64)
	sf.entry = make(map[float64]float64)
}

func (sf *SurvfuncRight) scanData() {

	var weight []float64
	if sf.weightpos != -1 {
		weight = sf.data[sf.weightpos]
	}

	var status []float64
	if sf.statuspos != -1 {
		status = sf.data[sf.statuspos]
	}

	var entry []float64
	if sf.entrypos != -1 {
		entry = sf.data[sf.entrypos]
	}

	for i, t := range sf.data[sf.timepos] {

		w := 1.0
		if sf.weightpos != -1 {
			w = weight[i]
		}

		if status == nil || status[i] == 1 {
			sf.events[t] += w
		}
		sf.total[t] += w

		if sf.entrypos != -1 {
			if entry[i] >= t {
				msg := fmt.Sprintf("Entry time %d is before the event/censoring time\n",
					i)
				os.Stderr.WriteString(msg)
				os.Exit(1)
			}
			sf.entry[entry[i]] += w
		}
	}
}

func (sf *SurvfuncRight) Time() []float64 {
	return sf.times
}

func rollback(x []float64) {
	var z float64
	for i := len(x) - 1; i >= 0; i-- {
		z += x[i]
		x[i] = z
	}
}

func (sf *SurvfuncRight) eventstats() {

	// Get the sorted distinct times (event or censoring)
	sf.times = make([]float64, len(sf.total))
	var i int
	for t := range sf.total {
		sf.times[i] = t
		i++
	}
	sort.Float64s(sf.times)

	// Get the weighted event count and risk set size at each time
	// point (in same order as Times).
	sf.nEvents = make([]float64, len(sf.times))
	sf.nRisk = make([]float64, len(sf.times))
	for i, t := range sf.times {
		sf.nEvents[i] = sf.events[t]
		sf.nRisk[i] = sf.total[t]
	}
	rollback(sf.nRisk)

	// Adjust for entry times
	if sf.entrypos != -1 {
		entry := make([]float64, len(sf.times))
		for t, w := range sf.entry {
			ii := sort.SearchFloat64s(sf.times, t)
			if t < sf.times[ii] {
				ii--
			}
			if ii >= 0 {
				entry[ii] += w
			}
		}
		rollback(entry)
		for i := 0; i < len(sf.nRisk); i++ {
			sf.nRisk[i] -= entry[i]
		}
	}
}

// compress removes times where no events occurred.
func (sf *SurvfuncRight) compress() {

	var ix []int
	for i := 0; i < len(sf.times); i++ {
		// Only retain events, except for the last point,
		// which is retained even if there are no events.
		if sf.nEvents[i] > 0 || i == len(sf.times)-1 {
			ix = append(ix, i)
		}
	}

	if len(ix) < len(sf.times) {
		for i, j := range ix {
			sf.times[i] = sf.times[j]
			sf.nEvents[i] = sf.nEvents[j]
			sf.nRisk[i] = sf.nRisk[j]
		}
		sf.times = sf.times[0:len(ix)]
		sf.nEvents = sf.nEvents[0:len(ix)]
		sf.nRisk = sf.nRisk[0:len(ix)]
	}
}

func (sf *SurvfuncRight) fit() {

	sf.survProb = make([]float64, len(sf.times))
	x := float64(1)
	for i := range sf.times {
		x *= 1 - sf.nEvents[i]/sf.nRisk[i]
		sf.survProb[i] = x
	}

	sf.survProbSE = make([]float64, len(sf.times))
	x = 0
	if sf.weightpos == -1 {
		for i := range sf.times {
			d := sf.nEvents[i]
			n := sf.nRisk[i]
			x += d / (n * (n - d))
			sf.survProbSE[i] = math.Sqrt(x) * sf.survProb[i]
		}
	} else {
		for i := range sf.times {
			d := sf.nEvents[i]
			n := sf.nRisk[i]
			x += d / (n * n)
			sf.survProbSE[i] = math.Sqrt(x)
		}
	}
}
