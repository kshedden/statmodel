// +build ignore

/*
This simulation generates data from a negative binomial distribution, and uses
the profile likelihood to estimate the dispersion parameter, and to produce
a confidence interval for it.
*/

package main

import (
	"fmt"
	"math"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel/glm"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	rng rand.Source
)

func genNegBinom(mean, disp float64) float64 {

	p := 1 - mean/(mean+disp*mean*mean)
	r := mean * (1 - p) / p

	g := distuv.Gamma{r, (1 - p) / p, rng}
	lam := g.Rand()

	po := distuv.Poisson{lam, rng}
	return po.Rand()
}

func simulate(n int, disp float64) dstream.Dstream {

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
		y[i] = genNegBinom(mn[i], disp)
	}

	return dstream.NewFromFlat([]interface{}{y, icept, x1, x2},
		[]string{"y", "icept", "x1", "x2"})
}

func main() {

	rng = rand.NewSource(4523745)

	n := 1000
	disp := 1.5
	data := simulate(n, disp)

	link := glm.NewLink(glm.LogLink)
	model := glm.NewGLM(data, "y").Family(glm.NewNegBinomFamily(disp, link)).Done()
	result := model.Fit()
	fmt.Printf("%v\n", result.Summary())

	nb := glm.NewNegBinomProfiler(result)

	dispMLE := nb.MLE()
	fmt.Printf("Dispersion MLE: %f\n", dispMLE)

	pct := 95
	lcb, ucb := nb.ConfInt(float64(pct) / 100)
	fmt.Printf("%d%% confidence interval: %f, %f\n", pct, lcb, ucb)
}
