package duration

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"

	"github.com/kshedden/statmodel/statmodel"
)

func TestSF1(t *testing.T) {

	var time []float64
	var status []float64
	n := 20

	for i := 0; i < n; i++ {
		time = append(time, float64(i))
		status = append(status, 1)
	}

	z := [][]float64{time, status}
	na := []string{"Time", "Status"}
	ds := statmodel.NewDataset(z, na)

	sf, err := NewSurvfuncRight(ds, "Time", "Status", nil)
	if err != nil {
		panic(err)
	}

	sf.Fit()

	// Check times and risk set sizes
	times := sf.Time()
	nrisk := sf.NumRisk()
	for i := 0; i < n; i++ {
		if times[i] != float64(i) {
			t.Fail()
		}
		if nrisk[i] != float64(n-i) {
			t.Fail()
		}
	}

	// From Python Statsmodels
	se := []float64{0.04873397, 0.06708204, 0.0798436, 0.08944272,
		0.09682458, 0.10246951, 0.10665365, 0.10954451,
		0.11124298, 0.1118034, 0.11124298, 0.10954451,
		0.10665365, 0.10246951, 0.09682458, 0.08944272,
		0.0798436, 0.06708204, 0.04873397}

	// Check probabilities and standard errors
	sp := sf.SurvProb()
	spse := sf.SurvProbSE()
	for i := 0; i < n; i++ {
		p := 1 - float64(i+1)/float64(n)
		if math.Abs(sp[i]-p) > 1e-6 {
			t.Fail()
		}

		if i < n-1 && math.Abs(spse[i]-se[i]) > 1e-6 {
			t.Fail()
		}
	}
}

func TestSF2(t *testing.T) {

	var time []float64
	var status []float64
	var weight []float64
	n := 20

	for i := 0; i < n; i++ {
		time = append(time, 10+float64(i))
		status = append(status, float64(i%2))
		weight = append(weight, float64(1+i%3))
	}

	z := [][]float64{time, status, weight}
	na := []string{"Time", "Status", "Weight"}
	ds := statmodel.NewDataset(z, na)

	cfg := &SurvfuncRightConfig{
		WeightVar: "Weight",
	}

	sf, err := NewSurvfuncRight(ds, "Time", "Status", cfg)
	if err != nil {
		panic(err)
	}

	sf.Fit()

	// Check times and risk set sizes
	times := sf.Time()
	for i := 0; i < 10; i++ {
		if times[i] != float64(11+2*i) {
			t.Fail()
		}
	}

	nriskExp := []float64{38, 33, 30, 26, 21, 18, 14, 9, 6, 2}
	nrisk := sf.NumRisk()
	if !floats.EqualApprox(nrisk, nriskExp, 1e-6) {
		t.Fail()
	}

	// From Python Statsmodels
	pr := []float64{0.94736842, 0.91866029, 0.82679426, 0.7631947, 0.7268521,
		0.60571008, 0.51918007, 0.46149339, 0.2307467, 0.}
	se := []float64{0.03721615, 0.04799287, 0.07507762, 0.09271045, 0.10422477,
		0.14185225, 0.17414403, 0.20657159, 0.35497205, 0.79120488}

	// Check probabilities and standard errors
	if !floats.EqualApprox(pr, sf.SurvProb(), 1e-6) {
		t.Fail()
	}
	if !floats.EqualApprox(se, sf.SurvProbSE(), 1e-6) {
		t.Fail()
	}
}

func TestSF3(t *testing.T) {

	var time []float64
	var status []float64
	var entry []float64
	n := 20

	for i := 0; i < n; i++ {
		time = append(time, 10+float64(i))
		status = append(status, float64(i%2))
		entry = append(entry, float64((10+i)/2))
	}

	z := [][]float64{time, status, entry}
	na := []string{"Time", "Status", "Entry"}
	data := statmodel.NewDataset(z, na)

	cfg := &SurvfuncRightConfig{
		EntryVar: "Entry",
	}

	sf, err := NewSurvfuncRight(data, "Time", "Status", cfg)
	if err != nil {
		panic(err)
	}

	sf.Fit()

	// Check times and risk set sizes
	times := sf.Time()
	if len(times) != 10 {
		t.Fail()
	}
	for i := 0; i < 10; i++ {
		if times[i] != float64(11+2*i) {
			t.Fail()
		}
	}

	// From Python Statsmodels
	nriskExp := []float64{11, 13, 15, 13, 11, 9, 7, 5, 3, 1}
	nrisk := sf.NumRisk()
	if !floats.EqualApprox(nrisk, nriskExp, 1e-6) {
		t.Fail()
	}

	// From Python Statsmodels
	pr := []float64{0.90909091, 0.83916084, 0.78321678, 0.72296934, 0.65724485,
		0.58421765, 0.50075798, 0.40060639, 0.26707092, 0}
	se := []float64{0.08667842, 0.10447861, 0.11148966, 0.11807514, 0.12429443,
		0.13018111, 0.13572541, 0.14076208, 0.14385416}

	// Check probabilities and standard errors
	if !floats.EqualApprox(sf.SurvProb(), pr, 1e-6) {
		t.Fail()
	}
	if !floats.EqualApprox(sf.SurvProbSE()[0:9], se[0:9], 1e-6) {
		t.Fail()
	}
}

func TestSF4(t *testing.T) {

	var time []float64
	var status []float64
	var entry []float64
	var weight []float64
	n := 20

	for i := 0; i < n; i++ {
		time = append(time, 10+float64(i))
		status = append(status, float64(i%2))
		entry = append(entry, float64((10+i)/2))
		weight = append(weight, float64(1+(i%3)))
	}

	z := [][]float64{time, status, entry, weight}
	na := []string{"Time", "Status", "Entry", "Weight"}
	data := statmodel.NewDataset(z, na)

	cfg := &SurvfuncRightConfig{
		EntryVar:  "Entry",
		WeightVar: "Weight",
	}

	sf, err := NewSurvfuncRight(data, "Time", "Status", cfg)
	if err != nil {
		panic(err)
	}

	sf.Fit()

	// Check times and risk set sizes
	times := sf.Time()
	if len(times) != 10 {
		t.Fail()
	}
	for i := 0; i < 10; i++ {
		if times[i] != float64(11+2*i) {
			t.Fail()
		}
	}

	// From Python Statsmodels
	nriskExp := []float64{23, 25, 30, 26, 21, 18, 14, 9, 6, 2}
	nrisk := sf.NumRisk()
	if !floats.EqualApprox(nrisk, nriskExp, 1e-6) {
		t.Fail()
	}

	// From Python Statsmodels
	pr := []float64{0.91304348, 0.87652174, 0.78886957, 0.72818729, 0.69351171,
		0.57792642, 0.4953655, 0.44032489, 0.22016245, 0.}
	se := []float64{0.06148755, 0.07335338, 0.09334908, 0.10803995, 0.11806865,
		0.1523137, 0.18276637, 0.21389069, 0.35928061, 0.79314725}

	// Check probabilities and standard errors
	if !floats.EqualApprox(sf.SurvProb(), pr, 1e-6) {
		t.Fail()
	}
	if !floats.EqualApprox(sf.SurvProbSE(), se, 1e-6) {
		t.Fail()
	}
}

func TestSF5(t *testing.T) {

	var time []float64
	var status []float64
	var entry []float64
	var weight []float64
	n := 20

	for i := 0; i < n; i++ {
		time = append(time, 10+float64(i/2))
		status = append(status, float64(i%2))
		entry = append(entry, float64((10+i)/2))
		weight = append(weight, float64(1+(i%3)))
	}

	z := [][]float64{time, status, entry, weight}
	na := []string{"Time", "Status", "Entry", "Weight"}
	data := statmodel.NewDataset(z, na)

	cfg := &SurvfuncRightConfig{
		EntryVar:  "Entry",
		WeightVar: "Weight",
	}

	sf, err := NewSurvfuncRight(data, "Time", "Status", cfg)
	if err != nil {
		panic(err)
	}

	sf.Fit()

	// Check times and risk set sizes
	times := sf.Time()
	if len(times) != 10 {
		t.Fail()
	}
	for i := 0; i < 10; i++ {
		if times[i] != float64(10+i) {
			t.Fail()
		}
	}

	// From Python Statsmodels
	nriskExp := []float64{19, 21, 20, 19, 21, 20, 15, 12, 8, 3}
	nrisk := sf.NumRisk()
	if !floats.EqualApprox(nriskExp, nrisk, 1e-6) {
		t.Fail()
	}

	// From Python Statsmodels
	pr := []float64{0.89473684, 0.85213033, 0.72431078, 0.64806754, 0.61720718,
		0.5246261, 0.45467595, 0.41678629, 0.26049143, 0.08683048}
	se := []float64{0.07443229, 0.08836142, 0.12372445, 0.14438804, 0.15203776,
		0.1749728, 0.19875706, 0.21551987, 0.30548946, 0.56173484}

	// Check probabilities and standard errors
	if !floats.EqualApprox(pr, sf.SurvProb(), 1e-6) {
		t.Fail()
	}
	if !floats.EqualApprox(se, sf.SurvProbSE(), 1e-6) {
		t.Fail()
	}
}
