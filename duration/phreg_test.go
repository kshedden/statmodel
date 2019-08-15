package duration

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/optimize"

	"github.com/kshedden/statmodel/statmodel"
)

func data1() statmodel.Dataset {

	da := [][]statmodel.Dtype{
		{1, 1, 2, 3, 3, 4},
		{1, 1, 0, 0, 1, 0},
		{4, 2, 5, 6, 6, 5},
	}

	varnames := []string{"Time", "Status", "X"}
	xnames := []string{"X"}

	return statmodel.NewDataset(da, varnames, "Time", xnames)
}

func data2() statmodel.Dataset {

	da := [][]statmodel.Dtype{
		{0, 1, 0, 1, 3, 2, 1, 2, 1, 3, 5},
		{1, 2, 4, 5, 4, 5, 6, 4, 6, 4, 8},
		{1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1},
		{4, 2, 3, 5, 1, 3, 5, 4, 2, 6, 6},
		{5, 2, 3, 1, 4, 2, 2, 5, 1, 8, 4},
		{1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2},
	}

	varnames := []string{"Entry", "Time", "Status", "X1", "X2", "Stratum"}
	xnames := []string{"X1", "X2"}

	return statmodel.NewDataset(da, varnames, "Time", xnames)
}

func data3() statmodel.Dataset {

	da := [][]statmodel.Dtype{
		{1, 1, 2, 3, 3, 4, 5, 5, 6, 7},
		{1, 1, 0, 0, 1, 0, 0, 1, 1, 1},
		{4, 2, 5, 6, 6, 5, 4, 3, 3, 5},
		{3, 2, 2, 0, 5, 4, 5, 6, 5, 4},
	}

	varnames := []string{"Time", "Status", "X1", "X2"}
	xnames := []string{"X1", "X2"}

	return statmodel.NewDataset(da, varnames, "Time", xnames)
}

func data4() statmodel.Dataset {

	da := [][]statmodel.Dtype{
		{1, 1, 2, 3, 3, 4, 5, 5, 6, 7},
		{1, 1, 0, 0, 1, 0, 0, 1, 1, 1},
		{4, 2, 5, 6, 6, 5, 4, 3, 3, 5},
		{3, 2, 2, 0, 5, 4, 5, 6, 5, 4},
	}

	varnames := []string{"Time", "Status", "X1", "X2"}
	xnames := []string{"X1", "X2"}

	return statmodel.NewDataset(da, varnames, "Time", xnames)
}

// Basic check, no strata, weights, or entry times.
func TestSimple(t *testing.T) {

	da := data1()
	ph := NewPHReg(da, "Status", nil)

	// Create an equivalent model that has L2 penalty weights all set to zero.
	config := DefaultPHRegConfig()
	config.L2Penalty = map[string]float64{"X1": 0, "X2": 0}
	phr := NewPHReg(da, "Status", config)

	for _, pq := range []*PHReg{ph, phr} {
		if fmt.Sprintf("%v", pq.stratumix) != "[[0 6]]" {
			t.Fail()
		}
		if fmt.Sprintf("%v", pq.etimes) != "[[1 3]]" {
			t.Fail()
		}
		if fmt.Sprintf("%v", pq.enter) != "[[[0 1 2 3 4 5] []]]" {
			t.Fail()
		}
		if fmt.Sprintf("%v", pq.exit) != "[[[0 1 2] [3 4]]]" {
			t.Fail()
		}
		if fmt.Sprintf("%v", pq.event) != "[[[0 1] [4]]]" {
			t.Fail()
		}
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

func TestStratified1(t *testing.T) {

	config := DefaultPHRegConfig()
	config.EntryVar = "Entry"
	config.StrataVar = "Stratum"

	da := data2()
	ph := NewPHReg(da, "Status", config)

	expected := "[[1 2 4 5] [4 6 8]]"
	if fmt.Sprintf("%v", ph.etimes) != expected {
		fmt.Printf("etimes do not match\n")
		fmt.Printf("Got      %v\n", ph.etimes)
		fmt.Printf("Expected %v\n", expected)
		t.Fail()
	}

	expected = "[[0 5] [5 11]]"
	if fmt.Sprintf("%v", ph.stratumix) != expected {
		fmt.Printf("Stratum boundaries do not match\n")
		fmt.Printf("Got      %v\n", ph.stratumix)
		fmt.Printf("Expected %v\n", expected)
		t.Fail()
	}

	expected = "[[[0 1 2 3] [] [4] []] [[5 6 7 8 9] [10] []]]"
	if fmt.Sprintf("%v", ph.enter) != expected {
		fmt.Printf("Entry times do not match\n")
		fmt.Printf("Got      %v\n", ph.enter)
		fmt.Printf("Expected %v\n", expected)
		t.Fail()
	}

	expected = "[[[0] [1] [2 4] [3]] [[5 7 9] [6 8] [10]]]"
	if fmt.Sprintf("%v", ph.exit) != expected {
		fmt.Printf("Exit times do not match\n")
		fmt.Printf("Got      %v\n", ph.exit)
		fmt.Printf("Expected %v\n", expected)
		t.Fail()
	}

	expected = "[[[0] [1] [4] [3]] [[7] [6 8] [10]]]"
	if fmt.Sprintf("%v", ph.event) != expected {
		fmt.Printf("Event times do not match\n")
		fmt.Printf("Got      %v\n", ph.event)
		fmt.Printf("Expected %v\n", expected)
		t.Fail()
	}

	ll := -26.950282147164277
	bl := ph.breslowLogLike([]float64{1, 2})
	if math.Abs(bl-ll) > 1e-5 {
		fmt.Printf("Breslow log-likelihood does not match\n")
		fmt.Printf("Got      %v\n", bl)
		fmt.Printf("Expected %v\n", ll)
		t.Fail()
	}

	ll = -32.44699788270529
	bl = ph.breslowLogLike([]float64{2, 1})
	if math.Abs(bl-ll) > 1e-5 {
		fmt.Printf("Breslow log-likelihood does not match\n")
		fmt.Printf("Got      %v\n", bl)
		fmt.Printf("Expected %v\n", ll)
		t.Fail()
	}

	score := make([]float64, 2)
	sc := []float64{-9.35565184, -8.0251037}
	ph.breslowScore([]float64{1, 2}, score)
	if !floats.EqualApprox(score, sc, 1e-5) {
		fmt.Printf("Breslow score does not match\n")
		fmt.Printf("Got      %v\n", score)
		fmt.Printf("Expected %v\n", sc)
		t.Fail()
	}

	sc = []float64{-13.5461984, -3.9178062}
	ph.breslowScore([]float64{2, 1}, score)
	if !floats.EqualApprox(score, sc, 1e-5) {
		fmt.Printf("Breslow score does not match\n")
		fmt.Printf("Got      %v\n", score)
		fmt.Printf("Expected %v\n", sc)
		t.Fail()
	}

	hess := make([]float64, 4)
	ph.breslowHess([]float64{1, 2}, hess)
	hs := []float64{-1.95989147, 1.23657039, 1.23657039, -1.13182375}
	if !floats.EqualApprox(hess, hs, 1e-5) {
		fmt.Printf("Breslow Hessian does not match\n")
		fmt.Printf("Got      %v\n", hess)
		fmt.Printf("Expected %v\n", hs)
		t.Fail()
	}

	ph.breslowHess([]float64{2, 1}, hess)
	hs = []float64{-1.12887225, 1.21185482, 1.21185482, -2.73825289}
	if !floats.EqualApprox(hess, hs, 1e-5) {
		fmt.Printf("Breslow Hessian does not match\n")
		fmt.Printf("Got      %v\n", hess)
		fmt.Printf("Expected %v\n", hs)
		t.Fail()
	}
}

func TestStratified2(t *testing.T) {

	var time, status, stratum, x1, x2 []statmodel.Dtype

	for i := 0; i < 100; i++ {
		x1 = append(x1, statmodel.Dtype(i%3))
		x2 = append(x2, statmodel.Dtype(i%7)-3)
		stratum = append(stratum, statmodel.Dtype(i%10))
		if i%5 == 0 {
			status = append(status, 0)
		} else {
			status = append(status, 1)
		}
		time = append(time, 10/statmodel.Dtype(4+i%3+i%7-3)+0.5*(statmodel.Dtype(i%6)-2))
	}

	da := [][]statmodel.Dtype{time, status, x1, x2, stratum}
	varnames := []string{"time", "status", "x1", "x2", "stratum"}
	xnames := []string{"x1", "x2"}

	data := statmodel.NewDataset(da, varnames, "time", xnames)

	c := DefaultPHRegConfig()
	c.StrataVar = "stratum"

	ph := NewPHReg(data, "status", c)
	result, err := ph.Fit()
	if err != nil {
		panic(err)
	}

	// Smoke test
	_ = result.Summary()

	par := result.Params()
	epar := []float64{0.1096391, 0.61394886}
	if !floats.EqualApprox(par, epar, 1e-5) {
		fmt.Printf("Parameter estimates differ:\n")
		fmt.Printf("Got      %v\n", par)
		fmt.Printf("Expected %v\n", epar)
		t.Fail()
	}

	se := result.StdErr()
	ese := []float64{0.17171136, 0.09304276}
	if !floats.EqualApprox(se, ese, 1e-5) {
		fmt.Printf("Standard errors differ:\n")
		fmt.Printf("Got      %v\n", se)
		fmt.Printf("Expected %v\n", ese)
		t.Fail()
	}
}

func TestPhregOptMethods(t *testing.T) {

	var time, status, stratum, x1, x2 []statmodel.Dtype

	for i := 0; i < 100; i++ {
		x1 = append(x1, statmodel.Dtype(i%3))
		x2 = append(x2, statmodel.Dtype(i%7)-3)
		stratum = append(stratum, statmodel.Dtype(i%10))
		if i%5 == 0 {
			status = append(status, 0)
		} else {
			status = append(status, 1)
		}
		time = append(time, 10/statmodel.Dtype(4+i%3+i%7-3)+0.5*(statmodel.Dtype(i%6)-2))
	}

	da := [][]statmodel.Dtype{time, status, x1, x2, stratum}
	varnames := []string{"time", "status", "x1", "x2", "stratum"}

	data := statmodel.NewDataset(da, varnames, "time", []string{"x1", "x2"})

	var par [][]float64
	var std [][]float64
	for _, m := range []optimize.Method{
		new(optimize.BFGS),
		new(optimize.LBFGS),
		new(optimize.CG),
		//new(optimize.Newton), //TODO
		new(optimize.GradientDescent),
		new(optimize.NelderMead),
	} {
		c := DefaultPHRegConfig()
		c.OptMethod = m
		c.StrataVar = "stratum"
		ph := NewPHReg(data, "status", c)
		result, err := ph.Fit()
		if err != nil {
			panic(err)
		}
		par = append(par, result.Params())
		std = append(std, result.StdErr())
	}

	// Compare eachj method to the first method
	for i := 1; i < len(par); i++ {
		if !floats.EqualApprox(par[0], par[i], 1e-6) {
			fmt.Printf("Parameter estimates differ:\n")
			fmt.Printf("Got       %v\n", par[i])
			fmt.Printf("Expected %v\n", par[0])
			t.Fail()
		}
		if !floats.EqualApprox(std[0], std[i], 1e-6) {
			fmt.Printf("Standard errors differ:\n")
			fmt.Printf("Got       %v\n", std[i])
			fmt.Printf("Expected %v\n", std[0])
			t.Fail()
		}
	}
}

func TestPhregRegularized(t *testing.T) {

	da := data3()
	pe := [][]float64{{-0.305179, 0}, {-0.145342, 0}, {0, 0}}

	for j, wt := range []float64{0.1, 0.2, 0.3} {

		c := DefaultPHRegConfig()
		c.L1Penalty = map[string]float64{"X1": wt, "X2": wt}
		ph := NewPHReg(da, "Status", c)
		rslt, err := ph.Fit()
		if err != nil {
			panic(err)
		}

		if !floats.EqualApprox(rslt.Params(), pe[j], 1e-5) {
			fmt.Printf("j=%d\nFound=%v\n", j, rslt.Params())
			fmt.Printf("Expected=%v\n", pe[j])
			t.Fail()
		}

		// Smoke test
		_ = rslt.Summary().String()
	}
}

func TestPhregFocus(t *testing.T) {

	da := data4()
	wt := 0.1

	c := DefaultPHRegConfig()
	c.L1Penalty = map[string]float64{"x1": wt, "x2": wt}
	c.L2Penalty = map[string]float64{"x1": wt, "x2": wt}

	ph := NewPHReg(da, "Status", c)

	phf := ph.Focus(0, []float64{1, 1}, nil)

	// The score at (1, 1) of the unprojected model
	score2d := make([]float64, 2)
	ph.Score(&PHParameter{[]float64{1, 1}}, score2d)

	// The score at 1 of the projected model
	score := make([]float64, 1)
	phf.Score(&PHParameter{[]float64{1}}, score)

	// Numerically calculate the score of the projected model
	dl := 1e-7
	scorenum := (phf.LogLike(&PHParameter{[]float64{1 + dl}}, false) -
		phf.LogLike(&PHParameter{[]float64{1}}, false)) / dl

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

	// data1 and data2 are equivalent after taking the weights into account
	da1 := [][]statmodel.Dtype{
		{1, 1, 2, 3, 3, 4},
		{1, 1, 0, 0, 1, 0},
		{4, 2, 5, 6, 6, 5},
		{1, 2, 1, 2, 1, 2},
	}
	varnames := []string{"Time", "Status", "X", "W"}
	data1 := statmodel.NewDataset(da1, varnames, "Time", []string{"X"})

	// "Unrolled" version of data1.
	da2 := [][]statmodel.Dtype{
		{1, 1, 1, 2, 3, 3, 3, 4, 4},
		{1, 1, 1, 0, 0, 0, 1, 0, 0},
		{4, 2, 2, 5, 6, 6, 6, 5, 5},
		{1, 1, 1, 1, 1, 1, 1, 1, 1},
	}
	data2 := statmodel.NewDataset(da2, varnames, "Time", []string{"X"})

	data3 := statmodel.NewDataset(da2[0:3], varnames[0:3], "Time", []string{"X"})

	c := DefaultPHRegConfig()
	c.WeightVar = "W"

	ph1 := NewPHReg(data1, "Status", c)
	ph2 := NewPHReg(data2, "Status", c)
	ph3 := NewPHReg(data3, "Status", nil)

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

	time := make([]statmodel.Dtype, n)
	status := make([]statmodel.Dtype, n)
	x := make([]statmodel.Dtype, n)

	// kw is the Weibull shape parameter.  The cumulative baseline hazard function
	// evaluated at time t is t^kw.
	for _, kw := range []float64{1, 2} {

		// Create a covariate, but there is no covariate effect in this test.
		for i := range x {
			x[i] = statmodel.Dtype(0.2 * rand.NormFloat64())
		}

		for i := range time {
			time[i] = statmodel.Dtype(math.Pow(-math.Log(rand.Float64()), 1/kw))
			t := statmodel.Dtype(math.Pow(-math.Log(rand.Float64()), 1/kw))
			if time[i] > t {
				time[i] = t
				status[i] = 0
			} else {
				status[i] = 1
			}
		}

		da := [][]statmodel.Dtype{time, status, x}

		varnames := []string{"time", "status", "x"}
		data := statmodel.NewDataset(da, varnames, "time", []string{"x"})

		model := NewPHReg(data, "status", nil)
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
			fmt.Printf("Got      %v\n", ra)
			fmt.Printf("Expected %v\n", kw)
			t.Fail()
		}
		if rd > 0.6 {
			fmt.Printf("Got      %v\n", rd)
			fmt.Printf("Expected < 0.6\n")
			t.Fail()
		}
	}
}
