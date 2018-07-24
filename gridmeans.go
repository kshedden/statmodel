package statmodel

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/dstream/formula"
	"gonum.org/v1/gonum/floats"
)

// advance takes a variable base representation of an integer and adds one to it.
// The arrays nvals and ix should have the same length, and the allowed values in
// ix[j] are 0, 1, ..., nvals[j]-1.
func advance(ix []int, nvals []int) bool {

	for j := range ix {
		if ix[j] < nvals[j]-1 {
			ix[j]++
			return false
		}
		ix[j] = 0
	}
	return true
}

// getMeans returns a map from variable names, to the mean value of the variable.
func getMeans(data dstream.Dstream) map[string]float64 {

	data.Reset()
	n := 0
	p := data.NumVar()
	vmean := make([]float64, p)
	for data.Next() {
		for j := 0; j < p; j++ {
			z := data.GetPos(j).([]float64)
			vmean[j] += floats.Sum(z)
			if j == 0 {
				n += len(z)
			}
		}
	}

	floats.Scale(1/float64(n), vmean)

	means := make(map[string]float64)
	names := data.Names()
	for j := range names {
		means[names[j]] = vmean[j]
	}

	return means
}

// GMrecord represents one grid point mean, i.e. a mean value at a specific
// point in the covariate space, calculated from a fitted model.
type GMrecord struct {
	Name GMname
	Mean float64
	SE   float64
}

type byMean []GMrecord

func (a byMean) Len() int           { return len(a) }
func (a byMean) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byMean) Less(i, j int) bool { return a[i].Mean < a[j].Mean }

type GridMeanResult struct {
	Records []GMrecord
}

// namerec represents the name of a grid point.
type GMname struct {

	// The variable names
	names []string

	// The values of the variables
	vals []float64
}

func (gn *GMname) String(fm map[string]string) string {

	var b []string
	var ok bool
	for i := range gn.names {
		na := gn.names[i]
		f := "%.1f"
		if fm != nil {
			f, ok = fm[na]
			if !ok {
				f = "%.1f"
			}
		}
		b = append(b, fmt.Sprintf("%s="+f, na, gn.vals[i]))
	}

	return strings.Join(b, ", ")
}

// Summary returns a string summarizing the grid mean values.  fm is a map from
// variable names to Sprintf format strings for the corresponding variable values,
// it may be passed as nil and defaults will be used.  mnf and sef are Sprintf
// format strings for the fitted means and standard errors.
func (gmt *GridMeanResult) Summary(fm map[string]string, mnf, sef string) string {

	var labels []string
	mx := 0
	for _, r := range gmt.Records {
		a := r.Name.String(fm)
		labels = append(labels, a)
		if len(a) > mx {
			mx = len(a)
		}
	}

	if mnf == "" {
		mnf = "%12.3f"
	}

	if sef == "" {
		sef = "%12.3f"
	}

	// fw returns the width from a Sprintf format string
	fw := func(x string) int {
		x = x[1:]
		v := strings.Split(x, ".")
		w, err := strconv.Atoi(v[0])
		if err != nil {
			panic(err)
		}
		return w
	}

	// Column widths
	mnw := fw(mnf)
	sew := fw(sef)

	tpx := fmt.Sprintf("%%-%ds %s %s", mx, mnf, sef)

	var s []string
	h := fmt.Sprintf(fmt.Sprintf("%%-%ds %%%ds %%%ds", mx, mnw, sew), "Group", "Mean  ", "SE  ")
	s = append(s, h)
	for i, gr := range gmt.Records {
		x := fmt.Sprintf(tpx, labels[i], gr.Mean, gr.SE)
		s = append(s, x)
	}

	return strings.Join(s, "\n")
}

// GridMeans constructs a table of fitted means and standard errors for the linear predictors
// of a fitted regression model.  The points argument is a map from variable names to an
// array of values at which the variable is to be fixed.  Variables that are not included
// in points are fixed at their mean values.  rslt is the fitted model used to
// produce the fitted values and standard errors, fml is the formula used to build the training
// set for the model, based on the data set "data".
func GridMeans(points map[string][]float64, fml string, rslt BaseResultser, data dstream.Dstream) *GridMeanResult {

	// Mean of each variable in the source data (prior to formula application)
	means := getMeans(data)

	// Source data names
	snames := data.Names()

	// arx contains a list of values to consider for each covariate.
	var arx [][]float64
	for _, na := range snames {
		v, ok := points[na]
		if ok {
			arx = append(arx, v)
		} else {
			arx = append(arx, []float64{means[na]})
		}
	}

	// The number of points to consider per covariate
	var np []int
	for _, v := range arx {
		np = append(np, len(v))
	}

	ix := make([]int, len(snames))
	dx := make([][]float64, len(snames))
	var rownames []GMname
	for {
		// Append one row
		var nr GMname
		for j := range dx {
			v := arx[j][ix[j]]
			dx[j] = append(dx[j], v)
			_, ok := points[snames[j]]
			if ok {
				nr.names = append(nr.names, snames[j])
				nr.vals = append(nr.vals, v)
			}
		}
		rownames = append(rownames, nr)

		if advance(ix, np) {
			break
		}
	}

	var da [][]interface{}
	for _, v := range dx {
		da = append(da, []interface{}{v})
	}
	df := dstream.NewFromArrays(da, snames)

	du := formula.New(fml, df).Done()

	fvals, se := predict(du, rslt)

	gmr := make([]GMrecord, len(fvals))
	for i := range fvals {
		gmr[i].Name = rownames[i]
		gmr[i].Mean = fvals[i]
		gmr[i].SE = se[i]
	}

	sort.Sort(byMean(gmr))

	return &GridMeanResult{gmr}
}

func predict(dx dstream.Dstream, rslt BaseResultser) ([]float64, []float64) {

	params := rslt.Params()
	names := rslt.Names()
	cov := rslt.VCov()
	p := len(names)

	var fvals []float64
	var se []float64
	for dx.Next() {
		var fv []float64
		var sev []float64

		var z [][]float64
		for _, na := range names {
			v := dx.Get(na).([]float64)
			z = append(z, v)
		}

		for j1 := range z {
			n := len(z[j1])
			if j1 == 0 {
				fv = make([]float64, n)
				sev = make([]float64, n)
			}
			for i := range z[j1] {
				fv[i] += params[j1] * z[j1][i]

				for j2 := 0; j2 < p; j2++ {
					sev[i] += z[j1][i] * cov[p*j1+j2] * z[j2][i]
				}
			}
		}

		for i := range se {
			se[i] = math.Sqrt(se[i])
		}

		fvals = append(fvals, fv...)
		se = append(se, sev...)
	}

	return fvals, se
}
