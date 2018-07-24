package statmodel

import (
	"fmt"
	"math"
	"testing"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/dstream/formula"
)

func TestMeans(t *testing.T) {

	da := data1()
	mn := getMeans(da)

	y := 1.1428571428571428
	x1 := 1.0
	x2 := 1.4285714285714286

	mnt := map[string]float64{"y": y, "x1": x1, "x2": x2}

	if len(mn) != len(mnt) {
		t.Fail()
	}

	for k := range mn {
		if math.Abs(mn[k]-mnt[k]) > 1e-8 {
			t.Fail()
		}
	}
}

func TestGM(t *testing.T) {

	da := data1()
	mn := getMeans(da)

	fml := "x1 + x2 + x1*x2"
	dbx := formula.New(fml, da).Done()
	db := dstream.MemCopy(dbx)

	model := &Mock{
		data: da,
		xpos: []int{0, 1, 2},
	}

	params := []float64{1, 2, 3}
	vcov := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}

	r := NewBaseResults(model, 0, params, db.Names(), vcov)

	pts := make(map[string][]float64)
	pts["x1"] = []float64{0, 1, 2}

	gmr := GridMeans(pts, fml, &r, da)

	_ = fmt.Sprintf("%s\n", gmr.Summary(nil, "%10.1f", "%10.1f"))

	for _, gm := range gmr.Records {
		var v float64
		switch {
		case gm.Name.vals[0] == 0:
			v = 0*params[0] + mn["x2"]*params[1] + 0*params[2]
		case gm.Name.vals[0] == 1:
			v = 1*params[0] + mn["x2"]*params[1] + mn["x2"]*params[2]
		case gm.Name.vals[0] == 2:
			v = 2*params[0] + mn["x2"]*params[1] + 2*mn["x2"]*params[2]
		default:
			t.Fail()
		}
		if math.Abs(v-gm.Mean) > 1e-8 {
			t.Fail()
		}
	}
}
