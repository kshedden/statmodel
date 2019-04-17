package duration

import (
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/kshedden/dstream/dstream"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// SurvfuncRight uses the method of Kaplan and Meier to estimate the
// survival distribution based on (possibly) right censored data.  The
// caller must set Data and TimeVar before calling the Fit method.
// StatusVar, WeightVar, and EntryVar are optional fields.
type SurvfuncRight struct {

	// The data used to perform the estimation.
	data dstream.Dstream

	// The name of the variable containing the minimum of the
	// event time and entry time.  The underlying data must have
	// float64 type.
	timeVar string

	// The name of a variable containing the status indicator,
	// which is 1 if the event occurred at the time given by
	// TimeVar, and 0 otherwise.  This is optional, and is assumed
	// to be identically equal to 1 if not present.
	statusVar string

	// The name of a variable containing case weights, optional.
	weightVar string

	// The name of a variable containing entry times, optional.
	entryVar string

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

	timepos   int
	statuspos int
	weightpos int
	entrypos  int
}

// NewSurvfuncRight creates a new value for fitting a survival function.
func NewSurvfuncRight(data dstream.Dstream, timevar, statusvar string) *SurvfuncRight {

	return &SurvfuncRight{
		data:      data,
		timeVar:   timevar,
		statusVar: statusvar,
	}
}

// Weight specifies the name of a case weight variable.
func (sf *SurvfuncRight) Weight(weight string) *SurvfuncRight {
	sf.weightVar = weight
	return sf
}

// Entry specifies the name of an entry time variable.
func (sf *SurvfuncRight) Entry(entry string) *SurvfuncRight {
	sf.entryVar = entry
	return sf
}

// Time returns the times at which the survival function changes.
func (sf *SurvfuncRight) Time() []float64 {
	return sf.times
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

	sf.data.Reset()

	sf.timepos = -1
	sf.statuspos = -1
	sf.weightpos = -1
	sf.entrypos = -1

	for k, na := range sf.data.Names() {
		switch na {
		case sf.timeVar:
			sf.timepos = k
		case sf.statusVar:
			sf.statuspos = k
		case sf.weightVar:
			sf.weightpos = k
		case sf.entryVar:
			sf.entrypos = k
		}
	}

	if sf.timepos == -1 {
		panic("Time variable not found")
	}
	if sf.statuspos == -1 {
		panic("Status variable not found")
	}
	if sf.weightVar != "" && sf.weightpos == -1 {
		panic("Status variable not found")
	}
	if sf.entryVar != "" && sf.entrypos == -1 {
		panic("Entry variable not found")
	}
}

func (sf *SurvfuncRight) scanData() {

	for j := 0; sf.data.Next(); j++ {

		time := sf.data.GetPos(sf.timepos).([]float64)

		var status []float64
		if sf.statuspos != -1 {
			status = sf.data.GetPos(sf.statuspos).([]float64)
		}

		var entry []float64
		if sf.entrypos != -1 {
			entry = sf.data.GetPos(sf.entrypos).([]float64)
		}

		var weight []float64
		if sf.weightpos != -1 {
			weight = sf.data.GetPos(sf.weightpos).([]float64)
		}

		for i, t := range time {

			w := float64(1)
			if sf.weightpos != -1 {
				w = weight[i]
			}

			if sf.statuspos == -1 || status[i] == 1 {
				sf.events[t] += w
			}
			sf.total[t] += w

			if sf.entrypos != -1 {
				if entry[i] >= t {
					msg := fmt.Sprintf("Entry time %d in chunk %d is before the event/censoring time\n",
						i, j)
					os.Stderr.WriteString(msg)
					os.Exit(1)
				}
				sf.entry[entry[i]] += w
			}
		}
	}
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
	sort.Sort(sort.Float64Slice(sf.times))

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

// Done indicates that the survival function has been configured and can now be fit.
func (sf *SurvfuncRight) Done() *SurvfuncRight {
	sf.init()
	sf.scanData()
	sf.eventstats()
	sf.compress()
	sf.fit()
	return sf
}

// SurvfuncRightPlotter is used to plot a survival function.
type SurvfuncRightPlotter struct {
	pts []plotter.XYs
	plt *plot.Plot

	labels []string

	lines []*plotter.Line

	width  vg.Length
	height vg.Length
}

// NewSurvfuncRightPlotter returns a default SurvfuncRightPlotter.
func NewSurvfuncRightPlotter() *SurvfuncRightPlotter {

	sp := &SurvfuncRightPlotter{
		width:  4,
		height: 4,
	}

	var err error
	sp.plt, err = plot.New()
	if err != nil {
		panic(err)
	}

	return sp
}

// Width sets the width of the survival function plot.
func (sp *SurvfuncRightPlotter) Width(w float64) *SurvfuncRightPlotter {
	sp.width = vg.Length(w)
	return sp
}

// Height sets the height of the survival function plot.
func (sp *SurvfuncRightPlotter) Height(h float64) *SurvfuncRightPlotter {
	sp.height = vg.Length(h)
	return sp
}

// Add plots a given survival function to the plot.
func (sp *SurvfuncRightPlotter) Add(sf *SurvfuncRight, label string) *SurvfuncRightPlotter {

	ti := sf.Time()
	pr := sf.SurvProb()

	m := len(ti)
	n := 2*m + 1

	pts := make(plotter.XYs, n)

	j := 0
	pts[j].X = 0
	pts[j].Y = 1
	j++

	for i := range ti {
		pts[j].X = ti[i]
		pts[j].Y = pts[j-1].Y
		j++
		pts[j].X = ti[i]
		pts[j].Y = pr[i]
		j++
	}

	sp.pts = append(sp.pts, pts)

	sp.labels = append(sp.labels, label)

	line, err := plotter.NewLine(pts)
	if err != nil {
		panic(err)
	}
	line.Color = plotutil.Color(len(sp.lines))
	sp.lines = append(sp.lines, line)

	return sp
}

// Plot constructs the plot.
func (sp *SurvfuncRightPlotter) Plot() *SurvfuncRightPlotter {

	sp.plt.Y.Min = 0
	sp.plt.Y.Max = 1

	sp.plt.X.Label.Text = "Time"
	sp.plt.Y.Label.Text = "Proportion alive"

	leg, err := plot.NewLegend()
	if err != nil {
		panic(err)
	}

	for i := range sp.lines {
		sp.plt.Add(sp.lines[i])
		leg.Add(sp.labels[i], sp.lines[i])
	}

	if len(sp.lines) > 1 {
		leg.Top = false
		leg.Left = true
		sp.plt.Legend = leg
	}

	return sp
}

// GetPlotStruct returns the plotting structure for this plot.
func (sp *SurvfuncRightPlotter) GetPlotStruct() *plot.Plot {
	return sp.plt
}

// Save writes the plot to the given file.
func (sp *SurvfuncRightPlotter) Save(fname string) {

	if err := sp.plt.Save(sp.width*vg.Inch, sp.height*vg.Inch, fname); err != nil {
		panic(err)
	}
}
