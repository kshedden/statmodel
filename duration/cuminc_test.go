package duration

import (
	"math"
	"testing"

	"github.com/kshedden/dstream/dstream"
)

func TestCI1(t *testing.T) {

	var time []float64
	var status []float64
	n := 20

	for i := 0; i < n; i++ {
		time = append(time, 10+float64(i/2))
		status = append(status, float64(i%2))
	}

	var z [][]interface{}
	z = append(z, []interface{}{time})
	z = append(z, []interface{}{status})
	na := []string{"Time", "Status"}
	data := dstream.NewFromArrays(z, na)

	ci := NewCumincRight(data, "Time", "Status").Done()

	// Check times
	for i := 0; i < 10; i++ {
		if ci.Times[i] != float64(10+i) {
			t.Fail()
		}
	}

	// From Python Statsmodels
	pr := []float64{0.05, 0.10277778, 0.15885417, 0.21893601, 0.28402468,
		0.35562221, 0.43616943, 0.53014119, 0.6476059, 0.82380295}
	se := []float64{0.04873397, 0.06891433, 0.08439262, 0.09743198, 0.10890472,
		0.11924923, 0.12870257, 0.13733875, 0.14476983, 0.14409121}

	// Check probabilities and standard errors
	for i, p := range ci.Probs[0] {
		if math.Abs(p-pr[i]) > 1e-6 {
			t.Fail()
		}

		if math.Abs(ci.ProbsSE[0][i]-se[i]) > 1e-6 {
			t.Fail()
		}
	}
}

func TestCI2(t *testing.T) {

	times := []float64{1, 1, 2, 4, 4, 4, 6, 6, 7, 8, 9, 9, 9, 1, 2, 2, 4, 4}
	stats := []float64{1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	var z [][]interface{}
	z = append(z, []interface{}{times})
	z = append(z, []interface{}{stats})
	na := []string{"Time", "Status"}
	data := dstream.NewFromArrays(z, na)

	ci := NewCumincRight(data, "Time", "Status").Done()

	// Check times
	ttimes := []float64{1, 2, 4, 6, 7}
	for i, s := range ttimes {
		if ci.Times[i] != s {
			t.Fail()
		}
	}

	// From Python Statsmodels
	pr := [][]float64{
		{0.11111111, 0.17037037, 0.17037037, 0.17037037, 0.17037037, 0.17037037, 0.17037037},
		{0., 0., 0.20740741, 0.20740741, 0.20740741, 0.20740741, 0.20740741},
		{0., 0., 0., 0.17777778, 0.26666667, 0.26666667, 0.26666667},
	}
	se := [][]float64{
		{0.07407407, 0.08976251, 0.08976251, 0.08976251, 0.08976251, 0.08976251, 0.08976251},
		{0., 0., 0.10610391, 0.10610391, 0.10610391, 0.10610391, 0.10610391},
		{0., 0., 0., 0.11196147, 0.12787781, 0.12787781, 0.12787781},
	}

	// Check probabilities and standard errors
	for j := 0; j < 2; j++ {
		for i, p := range ci.Probs[j] {
			if math.Abs(p-pr[j][i]) > 1e-6 {
				t.Fail()
			}
		}

		for i, s := range ci.ProbsSE[j] {
			if math.Abs(s-se[j][i]) > 1e-6 {
				print("A\n")
				t.Fail()
			}
		}
	}
}

func TestCI3(t *testing.T) {

	var time []float64
	var status []float64

	for i := 0; i < 20; i++ {
		time = append(time, 10+float64(i/2))
		status = append(status, float64(i%3))
	}

	var z [][]interface{}
	z = append(z, []interface{}{time})
	z = append(z, []interface{}{status})
	na := []string{"Time", "Status"}
	data := dstream.NewFromArrays(z, na)

	ci := NewCumincRight(data, "Time", "Status").Done()

	// Check times
	for i := 0; i < 10; i++ {
		if ci.Times[i] != float64(10+i) {
			t.Fail()
		}
	}

	// From Python Statsmodels
	pr := [][]float64{
		{0.05, 0.05, 0.10607639, 0.16215278, 0.16215278, 0.22897714, 0.2958015, 0.2958015, 0.3932537, 0.4907059},
		{0., 0.05277778, 0.10885417, 0.10885417, 0.16960359, 0.23642795, 0.23642795, 0.31438971, 0.41184191, 0.41184191},
	}
	se := [][]float64{
		{0.04873397, 0.04873397, 0.07114208, 0.08597339, 0.08597339, 0.10174694, 0.11255895, 0.11255895, 0.13106522, 0.13753272},
		{0., 0.05136219, 0.07274191, 0.07274191, 0.08957865, 0.10420542, 0.10420542, 0.11863944, 0.13371982, 0.13371982},
	}

	// Check probabilities and standard errors
	for j := 0; j < 2; j++ {
		for i, p := range ci.Probs[j] {
			if math.Abs(p-pr[j][i]) > 1e-6 {
				t.Fail()
			}
		}

		for i, s := range ci.ProbsSE[j] {
			if math.Abs(s-se[j][i]) > 1e-6 {
				print("A\n")
				t.Fail()
			}
		}
	}
}
