// +build ignore

/*
This simulation generates data from a Gamma GLM, computes the MLE of the
scale parameter, and constructs a profile likelihood confidence interval for it.
*/

package main

import (
	"fmt"
	"math"

	"golang.org/x/exp/rand"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel/glm"
	"gonum.org/v1/gonum/stat/distuv"
)

func simulate(n int, scale float64) dstream.Dstream {

	rng := rand.NewSource(4523745)

	x1 := make([]float64, n)
	x2 := make([]float64, n)
	for i := range x1 {
		x1[i] = rand.NormFloat64()
		x2[i] = rand.NormFloat64()
	}

	icept := make([]float64, n)
	for i := range icept {
		icept[i] = 1
	}

	mn := make([]float64, n)
	for i := range mn {
		mn[i] = math.Exp((x1[i] - x2[i]) / 2)
	}

	y := make([]float64, n)
	for i := range y {
		g := distuv.Gamma{1 / scale, scale * mn[i], rng}
		y[i] = g.Rand()
	}

	return dstream.NewFromFlat([]interface{}{y, icept, x1, x2},
		[]string{"y", "icept", "x1", "x2"})
}

func main() {

	scale := 3.0

	for _, n := range []int{500, 2000} {

		fmt.Printf("n=%d\n\n", n)
		data := simulate(1000, scale)

		model := glm.NewGLM(data, "y").Family(glm.NewFamily(glm.GammaFamily)).Link(glm.NewLink(glm.LogLink)).Done()
		result := model.Fit()
		fmt.Printf("%v\n", result.Summary())

		ps := glm.NewScaleProfiler(result)

		ps.ScaleMLE()
		fmt.Printf("Scale MLE: %f\n", ps.ScaleMLE())

		pct := 95
		lp, rp := ps.ConfInt(float64(pct) / 100)
		fmt.Printf("%d%% interval: %f, %f\n\n", pct, lp, rp)
	}
}
