package duration

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/optimize"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel/statmodel"
)

func data1() dstream.Dstream {

	data := `Time,Status,X
1,1,4
1,1,2
2,0,5
3,0,6
3,1,6
4,0,5
`

	tc := &dstream.CSVTypeConf{
		Float64: []string{"Time", "Status", "X"},
	}

	b := bytes.NewBuffer([]byte(data))
	da := dstream.FromCSV(b).TypeConf(tc).HasHeader().Done()
	da = dstream.MemCopy(da)

	return da
}

func data2() dstream.Dstream {
	data := `Entry,Time,Status,X1,X2,Stratum
0,1,1,4,5,1
1,2,1,2,2,1
0,4,0,3,3,1
1,5,1,5,1,1
3,4,1,1,4,1
2,5,0,3,2,2
1,6,1,5,2,2
2,4,1,4,5,2
1,6,1,2,1,2
3,4,0,6,8,2
5,8,1,6,4,2
`

	tc := &dstream.CSVTypeConf{
		Float64: []string{"Entry", "Time", "Status", "X1", "X2", "Stratum"},
	}

	b := bytes.NewBuffer([]byte(data))
	da := dstream.FromCSV(b).TypeConf(tc).HasHeader().Done()
	da = dstream.MemCopy(da)
	da = dstream.Convert(da, "Stratum", dstream.Uint64)
	da = dstream.Regroup(da, "Stratum", true)
	da = dstream.DropCols(da, "Stratum")
	return da
}

func data3() dstream.Dstream {
	data := `Time,Status,X1,X2
1,1,4,3
1,1,2,2
2,0,5,2
3,0,6,0
3,1,6,5
4,0,5,4
5,0,4,5
5,1,3,6
6,1,3,5
7,1,5,4
`

	tc := &dstream.CSVTypeConf{
		Float64: []string{"Time", "Status", "X1", "X2"},
	}

	b := bytes.NewBuffer([]byte(data))
	da := dstream.FromCSV(b).TypeConf(tc).HasHeader().Done()
	da = dstream.MemCopy(da)
	return da
}

func data4() dstream.Dstream {

	data := `Time,Status,X1,X2
1,1,4,3
1,1,2,2
2,0,5,2
3,0,6,0
3,1,6,5
4,0,5,4
5,0,4,5
5,1,3,6
6,1,3,5
7,1,5,4
`

	tc := &dstream.CSVTypeConf{
		Float64: []string{"Time", "Status", "X1", "X2"},
	}

	b := bytes.NewBuffer([]byte(data))
	da := dstream.FromCSV(b).TypeConf(tc).HasHeader().Done()
	da = dstream.MemCopy(da)

	return da
}

// Basic check, no strata, weights, or entry times.
func TestPhreg1(t *testing.T) {

	da := data1()
	ph := NewPHReg(da, "Time", "Status").Done()

	da.Reset()
	phr := NewPHReg(da, "Time", "Status").L2Weight([]float64{0}).Done()

	if fmt.Sprintf("%v", ph.enter) != "[[[0 1 2 3 4 5] []]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.exit) != "[[[0 1 2] [3 4]]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.event) != "[[[0 1] [4]]]" {
		t.Fail()
	}

	ll := -14.415134793348063
	for _, pq := range []*PHReg{ph, phr} {
		if math.Abs(pq.breslowLogLike([]float64{2})-ll) > 1e-5 {
			t.Fail()
		}
	}

	ll = -8.9840993267811093
	for _, pq := range []*PHReg{ph, phr} {
		if math.Abs(pq.breslowLogLike([]float64{1})-ll) > 1e-5 {
			t.Fail()
		}
	}

	score := make([]float64, 1)
	sc := -5.66698338
	for _, pq := range []*PHReg{ph, phr} {
		pq.breslowScore([]float64{2}, score)
		if math.Abs(score[0]-sc) > 1e-5 {
			t.Fail()
		}
	}

	sc = -5.09729328
	for _, pq := range []*PHReg{ph, phr} {
		pq.breslowScore([]float64{1}, score)
		if math.Abs(score[0]-sc) > 1e-5 {
			t.Fail()
		}
	}

	hv := -0.93879427
	hess := make([]float64, 1)
	for _, pq := range []*PHReg{ph, phr} {
		pq.breslowHess([]float64{1}, hess)
		if math.Abs(hess[0]-hv) > 1e-5 {
			t.Fail()
		}
	}
}

func TestPhreg2(t *testing.T) {

	da := data2()
	ph := NewPHReg(da, "Time", "Status").Entry("Entry").Done()

	if fmt.Sprintf("%v", ph.enter) != "[[[0 1 2 3] [] [4] []] [[0 1 2 3 4] [5] []]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.exit) != "[[[0] [1] [2 4] [3]] [[0 2 4] [1 3] [5]]]" {
		t.Fail()
	}
	if fmt.Sprintf("%v", ph.event) != "[[[0] [1] [4] [3]] [[2] [1 3] [5]]]" {
		t.Fail()
	}

	ll := -26.950282147164277
	if math.Abs(ph.breslowLogLike([]float64{1, 2})-ll) > 1e-5 {
		t.Fail()
	}

	ll = -32.44699788270529
	if math.Abs(ph.breslowLogLike([]float64{2, 1})-ll) > 1e-5 {
		t.Fail()
	}

	score := make([]float64, 2)
	sc := []float64{-9.35565184, -8.0251037}
	ph.breslowScore([]float64{1, 2}, score)
	if !floats.EqualApprox(score, sc, 1e-5) {
		t.Fail()
	}

	sc = []float64{-13.5461984, -3.9178062}
	ph.breslowScore([]float64{2, 1}, score)
	if !floats.EqualApprox(score, sc, 1e-5) {
		t.Fail()
	}

	hess := make([]float64, 4)
	ph.breslowHess([]float64{1, 2}, hess)
	hs := []float64{-1.95989147, 1.23657039, 1.23657039, -1.13182375}
	if !floats.EqualApprox(hess, hs, 1e-5) {
		t.Fail()
	}

	ph.breslowHess([]float64{2, 1}, hess)
	hs = []float64{-1.12887225, 1.21185482, 1.21185482, -2.73825289}
	if !floats.EqualApprox(hess, hs, 1e-5) {
		t.Fail()
	}
}

func TestPhreg3(t *testing.T) {

	var time, status, x1, x2 []float64
	var stratum []uint64

	for i := 0; i < 100; i++ {
		x1 = append(x1, float64(i%3))
		x2 = append(x2, float64(i%7)-3)
		stratum = append(stratum, uint64(i%10))
		if i%5 == 0 {
			status = append(status, 0)
		} else {
			status = append(status, 1)
		}
		time = append(time, 10/float64(4+i%3+i%7-3)+0.5*(float64(i%6)-2))
	}

	dat := [][]interface{}{[]interface{}{time}, []interface{}{status},
		[]interface{}{x1}, []interface{}{x2}, []interface{}{stratum}}
	na := []string{"time", "status", "x1", "x2", "stratum"}
	da := dstream.NewFromArrays(dat, na)
	da = dstream.Regroup(da, "stratum", true)
	da = dstream.DropCols(da, "stratum")

	ph := NewPHReg(da, "time", "status").Done()
	result, err := ph.Fit()
	if err != nil {
		panic(err)
	}

	// Smoke test
	_ = result.Summary()

	par := result.Params()
	if !floats.EqualApprox(par, []float64{0.1096391, 0.61394886}, 1e-5) {
		t.Fail()
	}

	se := result.StdErr()
	if !floats.EqualApprox(se, []float64{0.17171136, 0.09304276}, 1e-5) {
		t.Fail()
	}
}

func TestPhregMethods(t *testing.T) {

	var time, status, x1, x2 []float64
	var stratum []uint64

	for i := 0; i < 100; i++ {
		x1 = append(x1, float64(i%3))
		x2 = append(x2, float64(i%7)-3)
		stratum = append(stratum, uint64(i%10))
		if i%5 == 0 {
			status = append(status, 0)
		} else {
			status = append(status, 1)
		}
		time = append(time, 10/float64(4+i%3+i%7-3)+0.5*(float64(i%6)-2))
	}

	dat := [][]interface{}{[]interface{}{time}, []interface{}{status},
		[]interface{}{x1}, []interface{}{x2}, []interface{}{stratum}}
	na := []string{"time", "status", "x1", "x2", "stratum"}
	da := dstream.NewFromArrays(dat, na)
	da = dstream.Regroup(da, "stratum", true)
	da = dstream.DropCols(da, "stratum")

	var par [][]float64
	var std [][]float64
	for _, m := range []optimize.Method{
		new(optimize.BFGS),
		new(optimize.LBFGS),
		new(optimize.CG),
		//new(optimize.Newton),
		new(optimize.GradientDescent),
		new(optimize.NelderMead),
	} {
		ph := NewPHReg(da, "time", "status").Optimizer(m).Done()
		result, err := ph.Fit()
		if err != nil {
			panic(err)
		}
		par = append(par, result.Params())
		std = append(std, result.StdErr())
	}

	for i := 1; i < len(par); i++ {
		if !floats.EqualApprox(par[0], par[i], 1e-6) {
			t.Fail()
		}
		if !floats.EqualApprox(std[0], std[i], 1e-6) {
			t.Fail()
		}
	}
}

// Test whether the results are the same whether we scale or do not
// scale the covariates.
func TestPhregScaling(t *testing.T) {

	da := data2()
	ph1 := NewPHReg(da, "Time", "Status").Entry("Entry").Done()
	ph2 := NewPHReg(da, "Time", "Status").Entry("Entry").CovariateScale(statmodel.L2Norm).Done()

	r1, err := ph1.Fit()
	if err != nil {
		panic(err)
	}

	r2, err := ph2.Fit()
	if err != nil {
		panic(err)
	}

	if !floats.EqualApprox(r1.Params(), r2.Params(), 1e-5) {
		t.Fail()
	}
	if !floats.EqualApprox(r1.StdErr(), r2.StdErr(), 1e-5) {
		t.Fail()
	}
}

func TestPhregRegularized(t *testing.T) {

	da := data3()
	pe := [][]float64{{-0.305179, 0}, {-0.145342, 0}, {0, 0}}

	for j, wt := range []float64{0.1, 0.2, 0.3} {

		l1wgts := []float64{wt, wt}
		ph := NewPHReg(da, "Time", "Status").L1Weight(l1wgts).Done()
		rslt, err := ph.Fit()
		if err != nil {
			panic(err)
		}

		if !floats.EqualApprox(rslt.Params(), pe[j], 1e-5) {
			fmt.Printf("j=%d\nFound=%v\n", j, rslt.Params())
			fmt.Printf("Expected=%v\n", pe[j])
			t.Fail()
		}
		_ = rslt.Summary().String()
	}
}

func TestPhregFocus(t *testing.T) {

	da := data4()
	wt := 0.1
	wgt := []float64{wt, wt}
	ph := NewPHReg(da, "Time", "Status").L1Weight(wgt).L2Weight(wgt).Done()

	phf := ph.GetFocusable()
	phf.Focus(0, []float64{1, 1}, wt)

	// The score at (1, 1) of the unprojected model
	score2d := make([]float64, 2)
	ph.Score(&PHParameter{[]float64{1, 1}}, score2d)

	// The score at 1 of the projected model
	score := make([]float64, 1)
	phf.Score(&PHParameter{[]float64{1}}, score)

	// Numerically calculate the score of the projected model
	dl := 1e-7
	scorenum := (phf.LogLike(&PHParameter{[]float64{1 + dl}}) - phf.LogLike(&PHParameter{[]float64{1}})) / dl

	// Compare the numeric and analytic scores
	if math.Abs(score2d[0]-scorenum) > 1e-5 {
		t.Fail()
	}
	if math.Abs(score[0]-scorenum) > 1e-5 {
		t.Fail()
	}

	// The Hessian at (1, 1) of the unprojected model
	hess2d := make([]float64, 4)
	ph.Hessian(&PHParameter{[]float64{1, 1}}, statmodel.ObsHess, hess2d)

	// The Hessian at 1 of the projected model
	hess := make([]float64, 1)
	phf.Hessian(&PHParameter{[]float64{1}}, statmodel.ObsHess, hess)

	// Numerically calculate the Hessian of the projected model
	score1 := make([]float64, 1)
	score2 := make([]float64, 1)
	phf.Score(&PHParameter{[]float64{1 + dl}}, score1)
	phf.Score(&PHParameter{[]float64{1}}, score2)
	hessnum := (score1[0] - score2[0]) / dl

	if math.Abs(hess2d[0]-hessnum) > 1e-5 {
		t.Fail()
	}
	if math.Abs(hess[0]-hessnum) > 1e-5 {
		t.Fail()
	}
}

func TestWeights(t *testing.T) {

	data1 := `Time,Status,X,W
1,1,4,1
1,1,2,2
2,0,5,1
3,0,6,2
3,1,6,1
4,0,5,2
`
	data2 := `Time,Status,X,W
1,1,4,1
1,1,2,1
1,1,2,1
2,0,5,1
3,0,6,1
3,0,6,1
3,1,6,1
4,0,5,1
4,0,5,1
`

	tc := &dstream.CSVTypeConf{
		Float64: []string{"Time", "Status", "X", "W"},
	}

	b := bytes.NewBuffer([]byte(data1))
	da1 := dstream.FromCSV(b).TypeConf(tc).HasHeader().Done()
	da1 = dstream.MemCopy(da1)

	b = bytes.NewBuffer([]byte(data2))
	da2 := dstream.FromCSV(b).TypeConf(tc).HasHeader().Done()
	da2 = dstream.MemCopy(da2)

	ph1 := NewPHReg(da1, "Time", "Status").Weight("W").Done()
	ph2 := NewPHReg(dstream.DropCols(da2, "W"), "Time", "Status").Done()
	ph3 := NewPHReg(da2, "Time", "Status").Weight("W").Done()

	rslt1, err := ph1.Fit()
	if err != nil {
		panic(err)
	}

	rslt2, err := ph2.Fit()
	if err != nil {
		panic(err)
	}

	rslt3, err := ph3.Fit()
	if err != nil {
		panic(err)
	}

	if !floats.EqualApprox(rslt1.Params(), rslt2.Params(), 1e-5) {
		t.Fail()
	}

	if !floats.EqualApprox(rslt1.StdErr(), rslt2.StdErr(), 1e-5) {
		t.Fail()
	}

	if !floats.EqualApprox(rslt2.Params(), rslt3.Params(), 1e-5) {
		t.Fail()
	}

	if !floats.EqualApprox(rslt2.StdErr(), rslt3.StdErr(), 1e-5) {
		t.Fail()
	}
}

func TestBaselineHaz(t *testing.T) {

	n := 10000
	rand.Seed(3909)

	// kw is the Weibull shape parameter.  The cumulative baseline hazard function
	// evaluated at time t is t^kw.
	for _, kw := range []float64{1, 2} {

		x := make([]float64, n)
		tim := make([]float64, n)
		evt := make([]float64, n)

		// Create a covariate, but there is no covariate effect in this test.
		for i := range x {
			x[i] = 0.2 * rand.NormFloat64()
		}

		for i := range tim {
			tim[i] = math.Pow(-math.Log(rand.Float64()), 1/kw)
			t := math.Pow(-math.Log(rand.Float64()), 1/kw)
			if tim[i] > t {
				tim[i] = t
			} else {
				evt[i] = 1
			}
		}

		ar := make([][]interface{}, 3)
		ar[0] = []interface{}{tim}
		ar[1] = []interface{}{evt}
		ar[2] = []interface{}{x}

		df := dstream.NewFromArrays(ar, []string{"tim", "evt", "x"})

		model := NewPHReg(df, "tim", "evt").Done()
		result, err := model.Fit()
		if err != nil {
			panic(err)
		}

		ti, bch := model.BaselineCumHaz(0, result.Params())

		// The ratios below should cluster around kw.
		var ra, rd float64
		for i := 1; i < len(bch); i++ {
			r := math.Log(bch[i]) / math.Log(ti[i])
			ra += r
			rd += math.Abs(r - kw)
		}
		ra /= float64(len(bch))
		rd /= float64(len(bch))

		if math.Abs(ra-kw) > 0.07 {
			t.Fail()
		}
		if rd > 0.6 {
			t.Fail()
		}
	}
}
