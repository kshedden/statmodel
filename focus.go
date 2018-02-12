package statmodel

import (
	"fmt"

	"github.com/kshedden/dstream/dstream"
)

// FocusData is a Dstream that collapses a dataset by linear reduction
// over all but one covariate.
type FocusData struct {

	// The underlying data being focused
	data dstream.Dstream

	// The positions of the covariates in the underlying data
	xpos []int

	// The dependent variable positio
	ypos int

	// The position of an actual offset in data, if an actual
	// offset is present.  Otherwise this is -1.
	offsetPos int

	// The position of a weight variable in data, if a weight
	// variable is present.  Otherwise this is -1.
	weightPos int

	// The offset formed by combining the actual offset with the
	// effects of all non-focus variables.
	offset [][]float64

	// The DV (reference to data).
	y [][]float64

	// The focus x
	x [][]float64

	// Weights (optional)
	w [][]float64

	// Scale factors for the covariates
	xn []float64

	// Current chunk index
	chunk int
}

func (f *FocusData) Close() {
	// Do nothing
}

func (f *FocusData) NumObs() int {
	return f.data.NumObs()
}

func (f *FocusData) NumVar() int {
	nvar := 3
	if f.weightPos != -1 {
		nvar++
	}

	return nvar
}

func (f *FocusData) Reset() {
	f.chunk = -1
}

func (f *FocusData) Next() bool {
	f.chunk++
	return f.chunk < len(f.y)
}

func (f *FocusData) GetPos(pos int) interface{} {

	switch {
	case pos == 0:
		return f.y[f.chunk]
	case pos == 1:
		return f.x[f.chunk]
	case pos == 2:
		return f.offset[f.chunk]
	case pos == 3:
		if f.weightPos != -1 {
			return f.w[f.chunk]
		} else {
			panic("Invalid position in GetPos (no weight variable)\n")
		}
	default:
		panic("Invalid position in GetPos\n")
	}

	return nil
}

func (f *FocusData) Get(name string) interface{} {

	switch {
	case name == "y":
		return f.y[f.chunk]
	case name == "x":
		return f.x[f.chunk]
	case name == "off":
		return f.offset[f.chunk]
	case name == "wgt":
		if f.weightPos != -1 {
			return f.w[f.chunk]
		} else {
			panic("Invalid position in GetPos (no weight variable)\n")
		}
	default:
		msg := fmt.Sprintf("Name '%s' not found\n", name)
		panic(msg)
	}
}

func NewFocusData(data dstream.Dstream, xpos []int, ypos int, xn []float64) *FocusData {

	return &FocusData{
		data:      data,
		xpos:      xpos,
		ypos:      ypos,
		weightPos: -1,
		offsetPos: -1,
		chunk:     -1,
		xn:        xn,
	}
}

func (f *FocusData) Names() []string {

	na := []string{"y", "x", "off"}
	if f.weightPos != -1 {
		na = append(na, "wgt")
	}

	return na
}

func (f *FocusData) Offset(pos int) *FocusData {
	f.offsetPos = pos
	return f
}

func (f *FocusData) Weight(pos int) *FocusData {
	f.weightPos = pos
	return f
}

func (f *FocusData) Done() *FocusData {

	f.Reset()
	data := f.data

	data.Reset()
	for data.Next() {
		y := data.GetPos(f.ypos).([]float64)
		n := len(y)
		f.y = append(f.y, y)
		f.x = append(f.x, make([]float64, n))
		f.offset = append(f.offset, make([]float64, n))

		if f.weightPos != -1 {
			f.w = append(f.w, make([]float64, n))
		}
	}

	return f
}

func zero(x []float64) {
	for i := range x {
		x[i] = 0
	}
}

func (f *FocusData) Focus(fpos int, coeff []float64) {

	data := f.data
	data.Reset()

	for c := 0; data.Next(); c++ {

		if f.offsetPos != -1 {
			copy(f.offset[c], data.GetPos(f.offsetPos).([]float64))
		} else {
			zero(f.offset[c])
		}

		for j, k := range f.xpos {
			z := data.GetPos(k).([]float64)
			if j != fpos {
				for i, u := range z {
					f.offset[c][i] += u * coeff[j] / f.xn[j]
				}
			} else {
				for i, u := range z {
					f.x[c][i] = u / f.xn[j]
				}
			}
		}

		if f.weightPos != -1 {
			copy(f.w[c], data.GetPos(f.weightPos).([]float64))
		}
	}
}
