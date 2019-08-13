// Test GLM log-likelihood and score functions using numeric derivatives.  The
// tests confirm that the analytic score function agrees with the numeric
// derivative of the log-likelihood function.

package glm

import (
	"fmt"
	"testing"

	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
)

// A test problem
type difftestprob struct {
	title  string
	family *Family
	data   statmodel.Dataset
	weight bool
	offset bool
	params [][]float64
	scale  float64
	l2wgt  map[string]float64
}

var diffTests []difftestprob = []difftestprob{
	{
		title:  "Gaussian 1",
		family: NewFamily(GaussianFamily),
		data:   data1(false),
		weight: false,
		scale:  2,
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
	},
	{
		title:  "Gaussian 2",
		family: NewFamily(GaussianFamily),
		data:   data1(true),
		weight: true,
		scale:  2,
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
	},
	{
		title:  "Poisson 1",
		family: NewFamily(PoissonFamily),
		data:   data1(false),
		weight: false,
		scale:  1,
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
	},
	{
		title:  "Poisson 2",
		family: NewFamily(PoissonFamily),
		data:   data1(true),
		weight: true,
		scale:  1,
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
	},
	{
		title:  "Binomial 1",
		family: NewFamily(BinomialFamily),
		data:   data2(true),
		weight: true,
		params: [][]float64{{1, 0, 0}, {0, 1, 0}, {1, 1, 1}, {-1, 0, 1}},
		scale:  1,
	},
	{
		title:  "Gamma 1",
		family: NewFamily(GammaFamily),
		data:   data4(true),
		weight: true,
		params: [][]float64{{1, 0, 0}, {1, 1, 1}, {1, 0, -0.1}},
		scale:  2,
	},
	{
		title:  "Inverse Gaussian 1",
		family: NewFamily(InvGaussianFamily),
		data:   data4(true),
		params: [][]float64{{1, 0, 0}, {1, 1, 1}, {1, 0, -0.1}},
		weight: true,
		scale:  0.5,
	},
	{
		title:  "Tweedie 1",
		family: NewTweedieFamily(1.5, NewLink(LogLink)),
		data:   data1(false),
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
		weight: false,
		scale:  1.2,
	},
}

func TestGrad(t *testing.T) {

	for _, dt := range diffTests {

		config := DefaultConfig()

		if dt.weight {
			config.WeightVar = "w"
		}

		glm := NewGLM(dt.data, config)

		p := len(dt.params[0])
		ngrad := make([]float64, p)
		score := make([]float64, p)

		loglike := func(x []float64) float64 {
			return glm.LogLike(&GLMParams{x, dt.scale}, true)
		}

		for _, params := range dt.params {
			fd.Gradient(ngrad, loglike, params, nil)
			glm.Score(&GLMParams{params, dt.scale}, score)
			if !floats.EqualApprox(score, ngrad, 1e-5) {
				fmt.Printf("%s\n", dt.title)
				fmt.Printf("Numerical:  %v\n", ngrad)
				fmt.Printf("Analytical: %v\n", score)
				t.Fail()
			}
		}
	}
}
