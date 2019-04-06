package statmodel

import (
	"fmt"

	"github.com/kshedden/dstream/dstream"
)

// FocusData is a Dstream that collapses a dataset by linear reduction
// over all but one covariate.  Its main use is in coordinate
// optimization, e.g. for elastic net regression.
type FocusData struct {

	// The underlying data being focused
	data dstream.Dstream

	// The positions of the covariates in the underlying data
	xpos []int

	// The position of an actual offset in data, if an actual
	// offset is present.  Otherwise this is -1.
	offsetPos int

	// The names of all other variables from the dataset to retain
	// in the focused data.
	otherNames []string

	// The positions of the variables in otherNames
	otherPos []int

	// All other retained data, indexed by name
	otherData map[string][][]float64

	// The offset formed by combining the actual offset with the
	// effects of all non-focus variables.
	offset [][]float64

	// The focus x
	x [][]float64

	// Scale factors for the covariates
	xn []float64

	// Variable positions in the source data
	varpos map[string]int

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
	return 2 + len(f.otherNames)
}

func (f *FocusData) Reset() {
	f.chunk = -1
}

func (f *FocusData) Next() bool {
	f.chunk++
	return f.chunk < len(f.x)
}

func (f *FocusData) Get(na string) interface{} {

	switch {
	case na == "x":
		return f.x[f.chunk]
	case na == "off":
		return f.offset[f.chunk]
	default:
		z, ok := f.otherData[na]
		if !ok {
			msg := fmt.Sprintf("Variable '%s' not found\n", na)
			panic(msg)
		}
		return z[f.chunk]
	}
}

func (f *FocusData) GetPos(pos int) interface{} {

	switch {
	case pos == 0:
		return f.x[f.chunk]
	case pos == 1:
		return f.offset[f.chunk]
	default:
		return f.otherData[f.otherNames[pos-2]][f.chunk]
	}

	return nil
}

func NewFocusData(data dstream.Dstream, xpos []int, xn []float64) *FocusData {

	return &FocusData{
		data:      data,
		xpos:      xpos,
		offsetPos: -1,
		chunk:     -1,
		xn:        xn,
	}
}

func (f *FocusData) Names() []string {
	return append([]string{"x", "off"}, f.otherNames...)
}

// Offset sets the position of an offset variable, if one is present.
func (f *FocusData) Offset(pos int) *FocusData {
	f.offsetPos = pos
	return f
}

// Other provides the names of all additional variables to retain that
// are not covariates or offsets.
func (f *FocusData) Other(names []string) *FocusData {
	f.otherNames = names
	return f
}

func (f *FocusData) Done() *FocusData {

	f.Reset()
	data := f.data

	f.varpos = make(map[string]int)
	for k, v := range data.Names() {
		f.varpos[v] = k
	}
	for _, na := range f.otherNames {
		q, ok := f.varpos[na]
		if !ok {
			msg := fmt.Sprintf("Variable '%s' not found\n", na)
			panic(msg)
		}
		f.otherPos = append(f.otherPos, q)
	}

	f.otherData = make(map[string][][]float64)

	data.Reset()
	for data.Next() {

		// Get the length of the chunk
		x := data.GetPos(f.xpos[0]).([]float64)
		n := len(x)

		// Storage for the variable being focused on
		f.x = append(f.x, make([]float64, n))

		// Storage for the combined effects of all other
		// variables (and the actual offset if present).
		f.offset = append(f.offset, make([]float64, n))

		// Include all other retained variables (which are not
		// changed after each refocusing operation).
		for j, na := range f.otherNames {
			f.otherData[na] = append(f.otherData[na], data.GetPos(f.otherPos[j]).([]float64))
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

		// Set up the offset block
		if f.offsetPos != -1 {
			copy(f.offset[c], data.GetPos(f.offsetPos).([]float64))
		} else {
			zero(f.offset[c])
		}

		// Project into the offset block
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
	}
}
