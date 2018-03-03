package statmodel

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"

	"github.com/kshedden/dstream/dstream"
)

func TestKO1(t *testing.T) {

	x1 := []float64{3, 1, 5, 4, 2, 7, 5}
	x2 := []float64{4, 1, 2, 3, 8, 6, 2}
	x3 := []float64{2, 1, 5, 4, 9, 3, 1}
	x4 := []float64{1, 0, 3, 2, 5, 2, 4} // Don't knockoff this one

	da := dstream.NewFromArrays(
		[][]interface{}{[]interface{}{x1}, []interface{}{x2}, []interface{}{x3}, []interface{}{x4}},
		[]string{"x1", "x2", "x3", "x4"})

	ko := NewKnockoff(da, []string{"x1", "x2", "x3"})
	lmin := ko.lmin
	da = dstream.Dstream(ko)

	for r := 0; r < 3; r++ {

		if len(da.Names()) != 7 {
			t.Fail()
		}

		if da.NumVar() != 7 {
			t.Fail()
		}

		da.Reset()
		scale := make([]float64, 7)
		for da.Next() {

			for j := 0; j < 7; j++ {
				z := da.GetPos(j).([]float64)
				scale[j] += floats.Dot(z, z)
			}

			for i := 0; i < 3; i++ {
				for j := 0; j <= i; j++ {
					a1 := da.GetPos(i).([]float64)
					a2 := da.GetPos(j).([]float64)
					b1 := da.GetPos(4 + i).([]float64)
					b2 := da.GetPos(4 + j).([]float64)

					for _, x := range [][]float64{a1, a2, b1, b2} {
						u := floats.Dot(x, x)
						if math.Abs(u-1) > 1e-6 {
							t.Fail()
						}
					}

					if math.Abs(floats.Dot(a1, a2)-floats.Dot(b1, b2)) > 1e-6 {
						t.Fail()
					}

					d := math.Abs(floats.Dot(a1, b2) - floats.Dot(a1, a2))
					if i != j && d > 1e-6 {
						t.Fail()
					}

					if i == j && math.Abs(d-2*lmin) > 1e-6 {
						t.Fail()
					}
				}
			}
		}

		for _, j := range []int{0, 1, 2, 4, 5, 6} {
			if math.Abs(scale[j]-1) > 1e-6 {
				t.Fail()
			}
		}

		// Make sure it still works after copying out to
		// static arrays.
		if r == 1 {
			da.Reset()
			da = dstream.MemCopy(dstream.Dstream(da))
		}
	}
}

func TestKO2(t *testing.T) {

	x1 := []interface{}{[]float64{3, 1, 5, 4, 2, 7, 9}, []float64{6, 2, 1, 3, 4, 8, 5, 2}}
	x2 := []interface{}{[]float64{4, 1, 2, 3, 8, 6, 5}, []float64{6, 2, 3, 1, 5, 9, 3, 1}}
	x3 := []interface{}{[]float64{2, 1, 5, 4, 9, 3, 2}, []float64{9, 3, 5, 2, 1, 7, 5, 2}}
	x4 := []interface{}{[]float64{1, 0, 3, 2, 5, 2, 1}, []float64{3, 2, 1, 4, 2, 3, 4, 0}} // Don't knockoff this one

	da := dstream.NewFromArrays(
		[][]interface{}{x1, x2, x3, x4},
		[]string{"x1", "x2", "x3", "x4"})

	names := []string{"x1", "x2", "x3"}
	ko := NewKnockoff(da, names)
	lmin := ko.lmin
	da = dstream.Dstream(ko)

	// Need to cpoy this out because the results are random.
	da.Reset()
	da = dstream.MemCopy(da)

	for r := 0; r < 3; r++ {

		da.Reset()

		if len(da.Names()) != 7 {
			t.Fail()
		}

		if da.NumVar() != 7 {
			t.Fail()
		}

		scale := make([]float64, 7)
		for j := 0; j < 7; j++ {
			da.Reset()
			z := dstream.GetColPos(da, j).([]float64)
			scale[j] = floats.Dot(z, z)
		}

		for i := 0; i < 3; i++ {

			da.Reset()
			a1 := dstream.GetColPos(da, i).([]float64)
			da.Reset()
			b1 := dstream.GetColPos(da, 4+i).([]float64)

			for j := 0; j <= i; j++ {

				da.Reset()
				a2 := dstream.GetColPos(da, j).([]float64)
				da.Reset()
				b2 := dstream.GetColPos(da, 4+j).([]float64)

				if math.Abs(floats.Dot(a1, a2)-floats.Dot(b1, b2)) > 1e-6 {
					t.Fail()
				}

				d := math.Abs(floats.Dot(a1, b2) - floats.Dot(a1, a2))
				if i != j && d > 1e-6 {
					t.Fail()
				}

				if i == j && math.Abs(d-2*lmin) > 1e-6 {
					t.Fail()
				}
			}
		}

		for _, j := range []int{0, 1, 2, 4, 5, 6} {
			if math.Abs(scale[j]-1) > 1e-6 {
				t.Fail()
			}
		}
	}
}
