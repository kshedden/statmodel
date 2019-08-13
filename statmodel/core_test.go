package statmodel

import (
	"testing"

	"gonum.org/v1/gonum/floats"
)

func data1() ([]string, [][]Dtype) {
	x := [][]Dtype{
		[]Dtype{0, 1, 3, 2, 1, 1, 0},
		[]Dtype{1, 1, 1, 1, 1, 1, 1},
		[]Dtype{4, 1, -1, 3, 5, -5, 3},
	}
	return []string{"y", "x1", "x2"}, x
}

func data1b() ([]string, [][]Dtype) {
	x := [][]Dtype{
		[]Dtype{0, 1, 3, 2, 1, 1, 0},
		[]Dtype{1, 1, 1, 1, 1, 1, 1},
		[]Dtype{8, 2, -2, 6, 10, -10, 6},
	}
	return []string{"y", "x1", "x2"}, x
}

func data2() ([]string, [][]Dtype) {
	x := [][]Dtype{
		[]Dtype{0, 0, 1, 0, 1, 0, 0},
		[]Dtype{1, 1, 1, 1, 1, 1, 1},
		[]Dtype{4, 1, -1, 3, 5, -5, 3},
		[]Dtype{1, -1, 1, 1, 2, 5, -1},
	}
	return []string{"y", "x1", "x2", "x3"}, x
}

// A mock model for testing
type Mock struct {
	data [][]Dtype
	xpos []int
}

func (m *Mock) Dataset() [][]Dtype {
	return m.data
}

func (m *Mock) LogLike(params Parameter, exact bool) float64 {
	return 0
}

func (m *Mock) Score(params Parameter, score []float64) {
}

func (m *Mock) Hessian(params Parameter, ht HessType, score []float64) {
}

func (m *Mock) NumParams() int {
	return len(m.xpos)
}

func (m *Mock) NumObs() int {
	return len(m.data[0])
}

func (m *Mock) Xpos() []int {
	return m.xpos
}

func TestResult1(t *testing.T) {

	_, da := data1()
	model := &Mock{
		data: da,
		xpos: []int{1, 2},
	}

	params := []float64{1, 2}
	xnames := []string{"x1", "x2"}
	vcov := []float64{0, 0, 0, 0}

	r := NewBaseResults(model, 0, params, xnames, vcov)

	// Test fitted values on the training data.
	fv := []float64{9, 3, -1, 7, 11, -9, 7}
	if !floats.Equal(fv, r.FittedValues(nil)) {
		t.Fail()
	}

	// Test fitted values when passing a new data stream.
	_, da2 := data1b()
	fv = []float64{17, 5, -3, 13, 21, -19, 13}
	if !floats.Equal(fv, r.FittedValues(da2)) {
		t.Fail()
	}
}
