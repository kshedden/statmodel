// Test the PH regression log-likelihood and score functions using numeric
// derivatives.  The tests confirm that the analytic score function agrees
// with the numeric derivative of the log-likelihood function.

package duration

import (
	"fmt"
	"testing"

	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/floats"
)

const (
	tol = 1e-5
)

// A test problem
type difftestprob struct {
	title  string
	data   statmodel.Dataset
	xnames []string
	weight bool
	offset bool
	params [][]float64
	l2wgt  map[string]float64
}

var diffTests []difftestprob = []difftestprob{
	{
		title:  "",
		data:   data1(),
		xnames: []string{"x"},
		params: [][]float64{{0}, {1}, {-1}, {0.5}, {-0.5}},
	},
	{
		title:  "",
		data:   data2(),
		xnames: []string{"x1", "x2"},
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}, {-2, 1}},
	},
	{
		title:  "",
		data:   data3(),
		xnames: []string{"x1", "x2"},
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}, {2, -1}},
	},
	{
		title:  "",
		data:   data4(),
		xnames: []string{"x1", "x2"},
		params: [][]float64{{1, 0}, {0, 1}, {1, 1}, {-1, 1}, {-0.5, 1.3}},
	},
}

func TestGrad(t *testing.T) {

	for _, dt := range diffTests {

		config := DefaultPHRegConfig()

		model, err := NewPHReg(dt.data, "time", "status", dt.xnames, config)
		if err != nil {
			panic(err)
		}

		p := len(dt.params[0])
		ngrad := make([]float64, p)
		score := make([]float64, p)

		loglike := func(x []float64) float64 {
			return model.LogLike(&PHParameter{x}, true)
		}

		fdset := &fd.Settings{
			Formula: fd.Forward,
			Step:    1e-6,
		}

		for _, params := range dt.params {
			fd.Gradient(ngrad, loglike, params, fdset)
			model.Score(&PHParameter{params}, score)
			if !floats.EqualApprox(score, ngrad, tol) {
				fmt.Printf("%s\n", dt.title)
				fmt.Printf("Numerical:  %v\n", ngrad)
				fmt.Printf("Analytical: %v\n", score)
				t.Fail()
			}
		}
	}
}
