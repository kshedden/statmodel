package glm

import (
	"fmt"
	"testing"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
)

// A test problem
type difftestprob struct {
	title     string
	family    *Family
	data      dstream.Dstream
	weight    bool
	offset    bool
	params    [][]float64
	scale     float64
	l2wgt     map[string]float64
	scaletype []statmodel.ScaleType
}

var diffTests []difftestprob = []difftestprob{
	{
		title:     "Gaussian 1",
		family:    NewFamily(GaussianFamily),
		data:      data1(false),
		weight:    false,
		scale:     2,
		params:    [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
		scaletype: []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		title:     "Gaussian 2",
		family:    NewFamily(GaussianFamily),
		data:      data1(true),
		weight:    true,
		scale:     2,
		params:    [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
		scaletype: []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		title:     "Poisson 1",
		family:    NewFamily(PoissonFamily),
		data:      data1(false),
		weight:    false,
		scale:     1,
		params:    [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
		scaletype: []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		title:     "Poisson 2",
		family:    NewFamily(PoissonFamily),
		data:      data1(true),
		weight:    true,
		scale:     1,
		params:    [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
		scaletype: []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		title:     "Binomial 1",
		family:    NewFamily(BinomialFamily),
		data:      data2(true),
		weight:    true,
		params:    [][]float64{{1, 0, 0}, {0, 1, 0}, {1, 1, 1}, {-1, 0, 1}},
		scale:     1,
		scaletype: []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		title:     "Gamma 1",
		family:    NewFamily(GammaFamily),
		data:      data4(true),
		weight:    true,
		params:    [][]float64{{1, 0, 0}, {1, 1, 1}, {1, 0, -0.1}},
		scale:     2,
		scaletype: []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm, statmodel.Variance},
	},
	{
		title:     "Inverse Gaussian 1",
		family:    NewFamily(InvGaussianFamily),
		data:      data4(true),
		params:    [][]float64{{1, 0, 0}, {1, 1, 1}, {1, 0, -0.1}},
		weight:    true,
		scale:     0.5,
		scaletype: []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		title:     "Tweedie 1",
		family:    NewTweedieFamily(1.5, NewLink(LogLink)),
		data:      data1(false),
		params:    [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}},
		weight:    false,
		scale:     1.2,
		scaletype: []statmodel.ScaleType{statmodel.NoScale},
	},
}

func TestGrad(t *testing.T) {

	for _, dt := range diffTests {
		for _, scaletype := range dt.scaletype {

			glm := NewGLM(dt.data, "y")

			if dt.weight {
				glm = glm.Weight("w")
			}

			if dt.offset {
				glm = glm.Offset("off")
			}

			glm = glm.CovariateScale(scaletype)
			glm = glm.Family(dt.family).Done()

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
}
