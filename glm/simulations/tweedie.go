// +build ignore

/*
This simulation generates Tweedie data and uses the GLM likelihood to jointly estimate
the scale and variance power parameters.
*/

package main

import (
	"fmt"
	"math"

	"github.com/kshedden/statmodel/glm"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	rng rand.Source
)

// Generate a Tweedie random variable with mean mu and variance sig2 * mu^p.
func genTweedie(mu, p, sig2 float64) float64 {

	if p <= 1 || p >= 2 {
		panic("p must be between 1 and 2")
	}

	lam := math.Pow(mu, 2-p) / ((2 - p) * sig2)
	alp := (2 - p) / (p - 1)
	bet := math.Pow(mu, 1-p) / ((p - 1) * sig2)

	po := distuv.Poisson{lam, rng}
	n := po.Rand()

	var z float64
	for k := 0; k < int(n); k++ {
		g := distuv.Gamma{alp, bet, rng}
		z += g.Rand()
	}

	return z
}

type dataset struct {
	data     [][]float64
	varnames []string
	xnames   []string
}

func simulate(n int, pw, scale float64) dataset {

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
		y[i] = genTweedie(mn[i], pw, scale)
	}

	return dataset{
		data:     [][]float64{y, icept, x1, x2},
		varnames: []string{"y", "icept", "x1", "x2"},
		xnames:   []string{"icept", "x1", "x2"},
	}
}

func main() {
	rng = rand.NewSource(4523745)

	data := simulate(2000, 1.5, 2)

	link := glm.NewLink(glm.LogLink)

	c := glm.DefaultConfig()
	c.Family = glm.NewTweedieFamily(1.5, link)

	model := glm.NewGLM(data.data, data.varnames, "y", data.xnames, c)
	result := model.Fit()
	fmt.Printf("%v\n", result.Summary())

	tp := glm.NewTweedieProfiler(result)

	vpMLE := tp.VarPowerMLE()
	scaleMLE := tp.ScaleMLE()
	fmt.Printf("Variance power MLE: %f\n", vpMLE)
	fmt.Printf("Scale MLE:          %f\n", scaleMLE)
}
