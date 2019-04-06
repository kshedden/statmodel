package glm

import (
	"fmt"
	"testing"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/gonum/floats"
)

type ptlsh struct {
	family  *Family
	link    *Link
	alpha   float64
	data    dstream.Dstream
	weight  bool
	off     bool
	params  []float64
	ll      float64
	score   []float64
	exphess []float64
	obshess []float64
}

var pq = []ptlsh{
	{
		family:  NewFamily(PoissonFamily),
		weight:  false,
		data:    data1(false),
		params:  []float64{0, 0},
		ll:      -9.48490664979,
		score:   []float64{1, -6},
		exphess: []float64{-7, -10, -10, -86},
		obshess: []float64{-7, -10, -10, -86},
	},
	{
		family:  NewFamily(PoissonFamily),
		weight:  false,
		data:    data1(false),
		params:  []float64{1, 1},
		ll:      -659.930531049,
		score:   []float64{-661.4456244, -2940.68298198},
		exphess: []float64{-669.4456244, -2944.68298198, -2944.68298198, -13451.94403063},
		obshess: []float64{-669.4456244, -2944.68298198, -2944.68298198, -13451.94403063},
	},
	{
		family:  NewFamily(BinomialFamily),
		weight:  false,
		data:    data2(false),
		params:  []float64{0, 0, 0},
		ll:      -4.85203026392,
		score:   []float64{-1.5, -1, -1},
		exphess: []float64{-1.75, -2.5, -2, -2.5, -21.5, 3.25, -2, 3.25, -8.5},
		obshess: []float64{-1.75, -2.5, -2, -2.5, -21.5, 3.25, -2, 3.25, -8.5},
	},
	{
		family: NewFamily(BinomialFamily),
		link:   NewLink(LogLink),
		weight: true,
		data:   data2(true),
		params: []float64{-0.7, 0.1, 0},
		ll:     -14.070884019230451,
		score:  []float64{-12.99445525, -39.37101499, 2.18964978},
		exphess: []float64{-40.50897618, -144.25622765, -47.39149341,
			-144.25622765, -678.14114997, -178.31768404,
			-47.39149341, -178.31768404, -115.39745549},
		obshess: []float64{45.05775111, 654.03495551, 518.19721783,
			654.03495551, 4621.88794831, 2715.89536808,
			518.19721783, 2715.89536808, 941.69106211},
	},
	{
		family: NewFamily(BinomialFamily),
		weight: false,
		data:   data2(false),
		params: []float64{1, 0, 1},
		ll:     -11.818141431,
		score:  []float64{-3.59249274, -3.06001622, -5.53517637},
		exphess: []float64{-0.86262393, -1.84351226, 0.08233338, -1.84351226, -6.42091245,
			-0.02006538, 0.08233338, -0.02006538, -1.05735013},
		obshess: []float64{-0.86262393, -1.84351226, 0.08233338, -1.84351226, -6.42091245,
			-0.02006538, 0.08233338, -0.02006538, -1.05735013},
	},
	{
		family: NewFamily(BinomialFamily),
		weight: false,
		data:   data2(false),
		params: []float64{0, -1, 2},
		ll:     -16.8573417434,
		score:  []float64{-0.66377831, 7.25672511, -3.82448106},
		exphess: []float64{-0.59521913, -2.01281245, -0.68818286, -2.01281245, -8.51489657,
			-2.86562434, -0.68818286, -2.86562434, -1.18506228},
		obshess: []float64{-0.59521913, -2.01281245, -0.68818286, -2.01281245, -8.51489657,
			-2.86562434, -0.68818286, -2.86562434, -1.18506228},
	},
	{
		family: NewFamily(GammaFamily),
		weight: true,
		data:   data4(true),
		params: []float64{0.1, 0.1, 0.1},
		ll:     -43.463688316896253,
		score:  []float64{41.91666667, -141.75, 81.83333333},
		exphess: []float64{-844.11805556, 1256.1875, -1401.23611111,
			1256.1875, -8480.39583333, 7981.70833333,
			-1401.23611111, 7981.70833333, -8048.80555556},
		obshess: []float64{-844.11805556, 1256.1875, -1401.23611111,
			1256.1875, -8480.39583333, 7981.70833333,
			-1401.23611111, 7981.70833333, -8048.80555556},
	},
	{
		family: NewFamily(InvGaussianFamily),
		weight: true,
		data:   data4(true),
		params: []float64{0.1, 0.1, 0.1},
		ll:     -46.917965084595942,
		score:  []float64{-9.40831849, -32.75370535, -7.01395223},
		exphess: []float64{-70.37290533, 86.98514743, -112.07064966,
			86.98514743, -713.48807251, 625.27145184,
			-112.07064966, 625.27145184, -640.63104102},
		obshess: []float64{-70.37290533, 86.98514743, -112.07064966,
			86.98514743, -713.48807251, 625.27145184,
			-112.07064966, 625.27145184, -640.63104102},
	},
	{
		family: NewNegBinomFamily(1.5, NewLink(LogLink)),
		alpha:  1.5,
		weight: true,
		data:   data4(true),
		params: []float64{1, 0, -1},
		ll:     -77.310157634140779,
		score:  []float64{17.14149583, -23.34656954, 56.64897996},
		exphess: []float64{-6.54801803, -14.02138681, -0.8840382,
			-14.02138681, -50.90492947, -3.13023238,
			-0.8840382, -3.13023238, -8.54267285},
		obshess: []float64{-9.57814454, -24.11165106, -9.90658666,
			-24.11165106, -100.95443538, -20.30041455,
			-9.90658666, -20.30041455, -12.13755286},
	},
	{
		family: NewFamily(PoissonFamily),
		weight: true,
		data:   data5(true),
		off:    true,
		params: []float64{-1, 2},
		ll:     -10716.200029495829,
		score:  []float64{-10694.53706902, -49424.45601021},
		exphess: []float64{-10712.53706902, -49428.45601021,
			-49428.45601021, -233692.95149924},
		obshess: []float64{-10712.53706902, -49428.45601021,
			-49428.45601021, -233692.95149924},
	},
}

func TestLLScoreHess(t *testing.T) {

	for pj, ps := range pq {

		glm := NewGLM(ps.data, "y")

		glm = glm.Family(ps.family)

		if ps.link != nil {
			glm = glm.Link(ps.link)
		}
		if ps.weight {
			glm = glm.Weight("w")
		}
		if ps.off {
			glm = glm.Offset("off")
		}
		glm = glm.Done()

		m := glm.NumParams()
		score := make([]float64, m)
		hess := make([]float64, m*m)

		ll := glm.LogLike(&GLMParams{ps.params, 1})
		if !scalarClose(ll, ps.ll, 1e-5) {
			fmt.Printf("LogLike %d:\n", pj)
			t.Fail()
		}

		glm.Score(&GLMParams{ps.params, 1}, score)
		if !floats.EqualApprox(score, ps.score, 1e-5) {
			fmt.Printf("Score %d:\n", pj)
			t.Fail()
		}

		glm.Hessian(&GLMParams{ps.params, 1}, statmodel.ExpHess, hess)
		if !floats.EqualApprox(hess, ps.exphess, 1e-5) {
			fmt.Printf("Hessian %d:\n", pj)
			t.Fail()
		}

		glm.Hessian(&GLMParams{ps.params, 1}, statmodel.ObsHess, hess)
		if !floats.EqualApprox(hess, ps.obshess, 1e-5) {
			fmt.Printf("Hessian %d:\n", pj)
			t.Fail()
		}
	}
}
