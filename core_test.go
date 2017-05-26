package statmodel

import (
	"testing"

	"github.com/gonum/floats"
	"github.com/kshedden/dstream/dstream"
)

func data1() (dstream.Dstream, dstream.Reg) {
	y := []interface{}{
		[]float64{0, 1, 3, 2, 1, 1, 0},
	}
	x1 := []interface{}{
		[]float64{1, 1, 1, 1, 1, 1, 1},
	}
	x2 := []interface{}{
		[]float64{4, 1, -1, 3, 5, -5, 3},
	}
	dat := [][]interface{}{y, x1, x2}
	na := []string{"y", "x1", "x2"}
	da := dstream.NewFromArrays(dat, na)
	dr := dstream.NewReg(da, "y", []string{"x1", "x2"}, "", "")
	return da, dr
}

func data2() (dstream.Dstream, dstream.Reg) {
	y := []interface{}{
		[]float64{0, 0, 1, 0, 1, 0, 0},
	}
	x1 := []interface{}{
		[]float64{1, 1, 1, 1, 1, 1, 1},
	}
	x2 := []interface{}{
		[]float64{4, 1, -1, 3, 5, -5, 3},
	}
	x3 := []interface{}{
		[]float64{1, -1, 1, 1, 2, 5, -1},
	}
	dat := [][]interface{}{y, x1, x2, x3}
	na := []string{"y", "x1", "x2", "x3"}
	da := dstream.NewFromArrays(dat, na)
	dr := dstream.NewReg(da, "y", []string{"x1", "x2", "x3"}, "", "")
	return da, dr
}

func TestDims(t *testing.T) {

	da, dr := data1()
	if da.NumObs() != 7 || dr.NumObs() != 7 {
		t.Fail()
	}
	if da.NumVar() != 3 || dr.NumVar() != 3 {
		t.Fail()
	}
	if dr.NumCov() != 2 {
		t.Fail()
	}

	da, dr = data2()
	if da.NumObs() != 7 || dr.NumObs() != 7 {
		t.Fail()
	}
	if da.NumVar() != 4 || dr.NumVar() != 4 {
		t.Fail()
	}
	if dr.NumCov() != 3 {
		t.Fail()
	}
}

// A mock model for testing
type Mock struct {
	data dstream.Reg
}

func (m *Mock) DataSet() dstream.Reg {
	return m.data
}

func (m *Mock) LogLike(params []float64, scale float64) float64 {
	return 0
}

func (m *Mock) Score(params []float64, scale float64, score []float64) {
}

func (m *Mock) Hessian(params []float64, scale float64, ht HessType, score []float64) {
}

func TestFitParams1(t *testing.T) {

	_, dr := data1()
	model := &Mock{
		dr,
	}

	FitParams(model, make([]float64, dr.NumCov()))
}

func TestResult1(t *testing.T) {

	da, dr := data1()
	rd := dstream.NewReg(dr, "y", nil, "", "")
	model := &Mock{
		rd,
	}

	params := []float64{1, 2}
	xnames := []string{"x1", "x2"}
	vcov := []float64{0, 0, 0, 0}

	r := NewBaseResults(model, 0, params, xnames, vcov)

	fv := []float64{9, 3, -1, 7, 11, -9, 7}
	if !floats.Equal(fv, r.FittedValues(nil)) {
		t.Fail()
	}

	f := func(x interface{}) {
		z := x.([]float64)
		for i, _ := range z {
			z[i] = 2 * z[i]
		}
	}

	// Test when passing a new data stream.
	da = dstream.Mutate(da, "x2", f)
	dr = dstream.NewReg(da, "y", nil, "", "")
	fv = []float64{17, 5, -3, 13, 21, -19, 13}
	if !floats.Equal(fv, r.FittedValues(dr)) {
		t.Fail()
	}
}
