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
		na = append(na, "wt")
	}

	return dstream.NewFromArrays(da, na)
}

func TestFocus(t *testing.T) {

	for _, wt := range []bool{false, true} {

		da := fdata1(wt)

		fd := NewFocusData(da, []int{1, 2, 3}, []float64{1, 1, 1})

		othernames := []string{"y"}

		if wt {
			othernames = append(othernames, "wt")
		}

		fd = fd.Other(othernames).Done()

		fd.Focus(0, []float64{2, 3, 4})
		fd.Reset()
		fd.Next()

		if fd.NumObs() != 7 {
			t.Fail()
		}

		m := 3
		if wt {
			m++
		}
		if fd.NumVar() != m {
			t.Fail()
		}

		// x
		xe := []float64{1, 1, 1, 1, 1, 1, 1}
		if !floats.EqualApprox(fd.GetPos(0).([]float64), xe, 1e-6) {
			t.Fail()
		}
		if !floats.EqualApprox(fd.Get("x").([]float64), xe, 1e-6) {
			t.Fail()
		}

		// offset
		offe := []float64{24, 7, 5, 13, 31, 5, 33}
		if !floats.EqualApprox(fd.GetPos(1).([]float64), offe, 1e-6) {
			t.Fail()
		}
		if !floats.EqualApprox(fd.Get("off").([]float64), offe, 1e-6) {
			t.Fail()
		}

		// y
		ye := []float64{0, 1, 3, 2, 1, 1, 0}
		if !floats.EqualApprox(fd.GetPos(2).([]float64), ye, 1e-6) {
			t.Fail()
		}
		if !floats.EqualApprox(fd.Get("y").([]float64), ye, 1e-6) {
			t.Fail()
		}

		if !wt {
			continue
		}

		// weight
		wte := []float64{1, 2, 2, 3, 1, 3, 2}
		if !floats.EqualApprox(fd.GetPos(3).([]float64), wte, 1e-6) {
			t.Fail()
		}
		if !floats.EqualApprox(fd.Get("wt").([]float64), wte, 1e-6) {
			t.Fail()
		}
	}
}
