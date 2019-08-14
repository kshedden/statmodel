package duration

import (
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/floats"

	"github.com/kshedden/dstream/dstream"
)

// Concordance calculates the survival concordance of Uno et al.
// (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3079915).
type Concordance struct {

	// Truncate at this time horizon
	tau float64

	// The risk scores that are being assessed
	score []float64

	// Event or censoring time
	time []float64

	// Event status
	status []float64

	// Number of pairs to check, if using random sampling to
	// estimate the concordance
	npair int

	// The survival function for the censoring distribution
	sf *SurvfuncRight
}

// NewConcordance creates a new Concordance value with the given parameters.
func NewConcordance(time, status, score []float64) *Concordance {

	c := &Concordance{
		time:   time,
		status: status,
		score:  score,
		npair:  10000,
	}

	return c
}

// NumPair sets the number of pairs of observations sampled at random
// to estimate the concordance.
func (c *Concordance) NumPair(npair int) *Concordance {
	c.npair = npair
	return c
}

// Done signals that the Concordance value has been built and now can be fit.
func (c *Concordance) Done() *Concordance {

	// Sort everything by time
	ii := make([]int, len(c.time))
	time1 := make([]float64, len(c.time))
	statusr := make([]float64, len(c.time))
	status1 := make([]float64, len(c.time))
	score1 := make([]float64, len(c.time))
	copy(time1, c.time)
	floats.Argsort(time1, ii)
	ncens := 0.0
	for i, j := range ii {
		// We want the survival function for censoring
		statusr[i] = 1 - c.status[j]
		status1[i] = c.status[j]
		ncens += statusr[i]
	}
	for i, j := range ii {
		score1[i] = c.score[j]
	}

	// Get the survival function for censoring
	da := dstream.NewFromArrays([][]interface{}{{time1}, {statusr}},
		[]string{"Time", "Status"})
	c.sf = NewSurvfuncRight(da, "Time", "Status").Done()
	if ncens == 0 {
		// No censoring, create a censoring survival function
		// with P(T>t) = 1 for all t.
		c.sf.times = []float64{0, math.Inf(1)}
		c.sf.survProb = []float64{1, 1}
	}

	c.time = time1
	c.status = status1
	c.score = score1

	return c
}

// Concordance returns the concordace statistic, using the given truncation
// parameter.
func (c *Concordance) Concordance(trunc float64) float64 {

	npair := 10000
	n := len(c.time)

	jt := sort.SearchFloat64s(c.time, trunc)
	if jt <= 0 {
		panic("Not enough data below truncation point.\n")
	}

	time := c.time
	status := c.status
	score := c.score

	st := c.sf.Time()
	sp := c.sf.SurvProb()

	var numer, denom float64

	for i := 0; i < npair; i++ {

		// Find a pair to compare
		var j1, j2 int
		for {
			j1 = rand.Intn(n)
			if j1 >= jt {
				continue
			}
			j2 = rand.Intn(n)
			if j2 <= j1 {
				continue
			}
			if (time[j1] < time[j2]) && (status[j1] == 1) {
				break
			}
		}

		jj := sort.SearchFloat64s(st, time[j1])
		if jj == len(st) {
			jj -= 1
		}
		g := sp[jj]

		denom += 1 / (g * g)
		if score[j1] > score[j2] {
			numer += 1 / (g * g)
		}

	}

	return numer / denom
}
