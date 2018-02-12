package statmodel

import (
	"testing"

	"gonum.org/v1/gonum/floats"

	"github.com/kshedden/dstream/dstream"
)

func fdata1(wgt bool) dstream.Dstream {
	y := []interface{}{
		[]float64{0, 1, 3, 2, 1, 1, 0},
	}
	x1 := []interface{}{
		[]float64{1, 1, 1, 1, 1, 1, 1},
	}
	x2 := []interface{}{
		[]float64{4, 1, -1, 3, 5, -5, 3},
	}
	x3 := []interface{}{
		[]float64{3, 1, 2, 1, 4, 5, 6},
	}
	w := []interface{}{
		[]float64{1, 2, 2, 3, 1, 3, 2},
	}
	da := [][]interface{}{y, x1, x2, x3}
	na := []string{"y", "x1", "x2", "x3"}

	if wgt {
		da = append(da, w)
		na = append(na, "w")
	}

	return dstream.NewFromArrays(da, na)
}

func TestFocus(t *testing.T) {

	for _, wt := range []bool{false, true} {

		da := fdata1(wt)

		fd := NewFocusData(da, []int{1, 2, 3}, 0)

		if wt {
			fd = fd.Weight(4)
		}

		fd = fd.Done()

		fd.Focus(0, []float64{2, 3, 4})
		fd.Reset()
		fd.Next()

		if fd.NumObs() != 7 {
			t.Fail()
		}

		if wt && fd.NumVar() != 4 {
			t.Fail()
		}

		if !wt && fd.NumVar() != 3 {
			t.Fail()
		}

		// y
		if !floats.EqualApprox(fd.GetPos(0).([]float64), []float64{0, 1, 3, 2, 1, 1, 0}, 1e-6) {
			t.Fail()
		}

		// x
		if !floats.EqualApprox(fd.GetPos(1).([]float64), []float64{1, 1, 1, 1, 1, 1, 1}, 1e-6) {
			t.Fail()
		}

		// offset
		if !floats.EqualApprox(fd.GetPos(2).([]float64), []float64{24, 7, 5, 13, 31, 5, 33}, 1e-6) {
			t.Fail()
		}

		if !wt {
			continue
		}

		// weight
		if !floats.EqualApprox(fd.GetPos(3).([]float64), []float64{1, 2, 2, 3, 1, 3, 2}, 1e-6) {
			t.Fail()
		}
	}
}
