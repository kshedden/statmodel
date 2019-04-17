package statmodel

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"io"
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

// getMeans returns a map from variable names in a Dstream, to the mean
// value of the variable.
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

// GMrecord represents the fitted mean value of a regression model
// and its corresponding standard errors, taken at a specific point
// in the covariate space.
type GMrecord struct {

	// A description of the point where the mean applies
	Name GMname

	// The fitted mean
	Mean float64

	// The standard error of the fitted mean
	SE float64

	// The point in the covariate space where the mean applies
	Vec []float64
}

func (gr *GMrecord) String(tpl string, fm map[string]string) string {
	return fmt.Sprintf(tpl, gr.Name.String(fm), gr.Mean, gr.SE)
}

type byMean []*GMrecord

func (a byMean) Len() int           { return len(a) }
func (a byMean) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byMean) Less(i, j int) bool { return a[i].Mean < a[j].Mean }

// GridMeanResult is the result of a grid mean calculation.
type GridMeanResult struct {
	Records []*GMrecord
	p       int
	vcov    []float64
}

// Gname represents the name and location of a grid point.
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
func (gmr *GridMeanResult) Summary(fm map[string]string, mnf, sef string) string {

	var labels []string
	mx := 0
	for _, r := range gmr.Records {
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
	for _, gr := range gmr.Records {
		s = append(s, gr.String(tpx, fm))
	}

	return strings.Join(s, "\n")
}

// PairContrastRecord contains the results of statistically contrasting
// fitted points corresponding to two points in the covariate space of a
// regression model.
type PairContrastRecord struct {

	// The first point being contrasted
	Rec1 *GMrecord

	// The second point being contrasted
	Rec2 *GMrecord

	// The difference of fitted values at the two contrasted points
	Diff float64

	// The standard error of the difference
	SE float64
}

type PairContrastResult struct {

	// Each record corresponds to one pair of points to contrast
	Records []*PairContrastRecord

	// The number of covariates
	p int

	// The variance covariance matrix of the parameters
	vcov []float64
}

func (pc *PairContrastResult) init() {

	p := pc.p
	vcov := pc.vcov

	for _, r := range pc.Records {

		var va float64
		for j1 := 0; j1 < p; j1++ {
			q1 := r.Rec1.Vec[j1] - r.Rec2.Vec[j1]
			for j2 := 0; j2 < p; j2++ {
				q2 := r.Rec1.Vec[j2] - r.Rec2.Vec[j2]
				va += q1 * vcov[j1*p+j2] * q2
			}
		}

		r.Diff = r.Rec1.Mean - r.Rec2.Mean
		r.SE = math.Sqrt(va)
	}

}

func (pc *PairContrastResult) Summary() string {

	var buf bytes.Buffer
	for _, v := range pc.Records {
		io.WriteString(&buf, fmt.Sprintf("%v", v))
	}

	return string(buf.Bytes())
}

// PairContrast groups a GridMeanResult according to specified values of a given variable.
func (gm *GridMeanResult) PairContrast(v1, v2 float64, vname string) *PairContrastResult {

	qr := make(map[uint64][2]*GMrecord)

	for _, r := range gm.Records {

		var buf bytes.Buffer
		var pos int = -1
		for i, na := range r.Name.names {
			if na == vname {
				if r.Name.vals[i] == v1 {
					pos = 0
				} else if r.Name.vals[i] == v2 {
					pos = 1
				} else {
					// A value of the focus variable not being contrasted
					continue
				}
			} else {
				err := binary.Write(&buf, binary.LittleEndian, uint64(10000*r.Name.vals[i]))
				if err != nil {
					panic(err)
				}
			}
		}

		// Not a relevant record
		if pos == -1 {
			continue
		}

		ha := fnv.New64a()
		_, err := ha.Write(buf.Bytes())
		if err != nil {
			panic(err)
		}
		ky := ha.Sum64()

		v := qr[ky]
		v[pos] = r
		qr[ky] = v
	}

	var pc []*PairContrastRecord
	for _, v := range qr {
		pc = append(pc,
			&PairContrastRecord{
				Rec1: v[0],
				Rec2: v[1],
			})
	}

	pcr := &PairContrastResult{
		Records: pc,
		p:       gm.p,
		vcov:    gm.vcov,
	}

	pcr.init()

	return pcr
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

	fvals, se, dm := predict(du, rslt)

	gmr := make([]*GMrecord, len(fvals))
	for i := range fvals {
		r := new(GMrecord)
		r.Name = rownames[i]
		r.Mean = fvals[i]
		r.SE = se[i]
		r.Vec = dm[i]
		gmr[i] = r
	}

	sort.Sort(byMean(gmr))

	return &GridMeanResult{
		Records: gmr,
		p:       len(rslt.Params()),
		vcov:    rslt.VCov(),
	}
}

// transpose takes a dstream and returns an array of arrays dm, such that
// dm[i] is the i^th row of the dstream.
func transpose(dx dstream.Dstream) [][]float64 {

	p := dx.NumVar()
	var dm [][]float64

	dx.Reset()
	for dx.Next() {

		var z [][]float64
		for j := 0; j < p; j++ {

			x := dx.GetPos(j).([]float64)

			// Allocate
			if j == 0 {
				z = make([][]float64, len(x))
				for i := range x {
					z[i] = make([]float64, p)
				}
			}

			for i := range x {
				z[i][j] = x[i]
			}
		}

		dm = append(dm, z...)
	}

	return dm
}

// predict takes a Dstream and a compatible BaseResultser from a fitted model,
// and returns predicted values and standard errors for the model at each row
// of the Dstream.  The returned values are the fitted values, the standard
// errors, and the rows of the Dstream as float64 arrays.  The Dstream must
// exactly match the BaseResultser, i.e. the kth parameter from the BaseResultser
// must correspond to the kth column of the Dstream.
func predict(dx dstream.Dstream, rslt BaseResultser) ([]float64, []float64, [][]float64) {

	params := rslt.Params()
	names := rslt.Names()
	cov := rslt.VCov()
	p := len(names)

	dm := transpose(dx)

	fvals := make([]float64, len(dm))
	se := make([]float64, len(dm))

	for i, z := range dm {
		for j1 := 0; j1 < p; j1++ {
			fvals[i] += params[j1] * z[j1]
			for j2 := 0; j2 < p; j2++ {
				se[i] += z[j1] * cov[p*j1+j2] * z[j2]
			}
		}
		se[i] = math.Sqrt(se[i])
	}

	return fvals, se, dm
}
