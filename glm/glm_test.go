/*
This set of tests uses very small datasets.  The results for comparison are taken from SAS, Stata, and R.
*/

package glm

import (
	"log"
	"math"
	"os"
	"testing"

	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/gonum/floats"
)

func scalarClose(x, y, eps float64) bool {
	return math.Abs(x-y) <= eps
}

func data1(wgt bool) statmodel.Dataset {

	y := []statmodel.Dtype{0, 1, 3, 2, 1, 1, 0}
	x1 := []statmodel.Dtype{1, 1, 1, 1, 1, 1, 1}
	x2 := []statmodel.Dtype{4, 1, -1, 3, 5, -5, 3}
	data := [][]statmodel.Dtype{y, x1, x2}
	varnames := []string{"y", "x1", "x2"}
	xnames := []string{"x1", "x2"}

	if wgt {
		w := []statmodel.Dtype{1, 2, 2, 3, 1, 3, 2}
		data = append(data, w)
		varnames = append(varnames, "w")
	}

	return statmodel.NewDataset(data, varnames, "y", xnames)
}

func data2(wgt bool) statmodel.Dataset {

	y := []statmodel.Dtype{0, 0, 1, 0, 1, 0, 0}
	x1 := []statmodel.Dtype{1, 1, 1, 1, 1, 1, 1}
	x2 := []statmodel.Dtype{4, 1, -1, 3, 5, -5, 3}
	x3 := []statmodel.Dtype{1, -1, 1, 1, 2, 5, -1}
	data := [][]statmodel.Dtype{y, x1, x2, x3}
	varnames := []string{"y", "x1", "x2", "x3"}
	xnames := []string{"x1", "x2", "x3"}

	if wgt {
		w := []statmodel.Dtype{2, 1, 3, 3, 4, 2, 3}
		data = append(data, w)
		varnames = append(varnames, "w")
	}

	return statmodel.NewDataset(data, varnames, "y", xnames)
}

func data3(wgt bool) statmodel.Dataset {

	y := []statmodel.Dtype{1, 1, 1, 0, 0, 0, 0}
	x1 := []statmodel.Dtype{1, 1, 1, 1, 1, 1, 1}
	x2 := []statmodel.Dtype{0, 1, 0, 0, -1, 0, 1}
	data := [][]statmodel.Dtype{y, x1, x2}
	varnames := []string{"y", "x1", "x2"}
	xnames := []string{"x1", "x2"}

	if wgt {
		w := []statmodel.Dtype{3, 3, 2, 3, 1, 3, 2}
		data = append(data, w)
		varnames = append(varnames, "w")
	}

	return statmodel.NewDataset(data, varnames, "y", xnames)
}

func data4(wgt bool) statmodel.Dataset {

	y := []statmodel.Dtype{3, 1, 5, 4, 2, 3, 6}
	x1 := []statmodel.Dtype{1, 1, 1, 1, 1, 1, 1}
	x2 := []statmodel.Dtype{4, 1, -1, 3, 5, -5, 3}
	x3 := []statmodel.Dtype{1, -1, 1, 1, 2, 5, -1}
	data := [][]statmodel.Dtype{y, x1, x2, x3}
	varnames := []string{"y", "x1", "x2", "x3"}
	xnames := []string{"x1", "x2", "x3"}

	if wgt {
		w := []statmodel.Dtype{3, 3, 2, 3, 1, 3, 2}
		data = append(data, w)
		varnames = append(varnames, "w")
	}

	return statmodel.NewDataset(data, varnames, "y", xnames)
}

func data5(wgt bool) statmodel.Dataset {

	y := []statmodel.Dtype{0, 1, 3, 2, 1, 1, 0}
	x1 := []statmodel.Dtype{1, 1, 1, 1, 1, 1, 1}
	x2 := []statmodel.Dtype{4, 1, -1, 3, 5, -5, 3}
	off := []statmodel.Dtype{0, 0, 1, 1, 0, 0, 0}
	data := [][]statmodel.Dtype{y, x1, x2, off}
	varnames := []string{"y", "x1", "x2", "off"}
	xnames := []string{"x1", "x2"}

	if wgt {
		w := []statmodel.Dtype{1, 2, 2, 3, 1, 3, 2}
		data = append(data, w)
		varnames = append(varnames, "w")
	}

	return statmodel.NewDataset(data, varnames, "y", xnames)
}

// A test problem
type testprob struct {
	title      string
	family     *Family
	data       statmodel.Dataset
	weight     bool
	offset     bool
	start      []float64
	params     []float64
	stderr     []float64
	vcov       []float64
	ll         float64
	scale      float64
	l2wgt      map[string]float64
	l1wgt      map[string]float64
	fitmethods []string
	paramstol  float64
	stderrtol  float64
	vcovtol    float64
	logliketol float64
	scaletol   float64
}

var glmTests []testprob = []testprob{
	{
		title:      "Gaussian 1 weighted OLS",
		family:     NewFamily(GaussianFamily),
		start:      nil,
		data:       data1(true),
		weight:     true,
		params:     []float64{1.316285, -0.047555},
		stderr:     []float64{0.277652, 0.080877},
		vcov:       []float64{0.077091, -0.004205, -0.004205, 0.006541},
		ll:         -19.14926021670413,
		scale:      1.0414236578435769,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Gaussian 2 weighted OLS",
		family: NewFamily(GaussianFamily),
		start:  nil,
		data:   data2(true),
		weight: true,
		params: []float64{0.191194, 0.046013, 0.090639},
		stderr: []float64{0.199909, 0.044360, 0.082265},
		vcov: []float64{0.039963, -0.005955, -0.011730,
			-0.005955, 0.001968, 0.001831,
			-0.011730, 0.001831, 0.006768},
		ll:         -11.876495505764467,
		scale:      0.25882586275287583,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Gaussian 3 weighted OLS",
		family:     NewFamily(GaussianFamily),
		start:      nil,
		data:       data3(true),
		weight:     true,
		params:     []float64{0.418605, 0.220930},
		stderr:     []float64{0.13620, 0.22926},
		vcov:       []float64{0.018551, -0.012367, -0.012367, 0.052560},
		ll:         -11.862285137866323,
		scale:      0.26589147286821707,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Poisson 1 weighted MLE",
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data1(true),
		weight:     true,
		params:     []float64{0.266817, -0.035637},
		stderr:     []float64{0.236179, 0.067480},
		vcov:       []float64{0.055780, -0.001012, -0.001012, 0.004553},
		ll:         -19.00280708909699,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Poisson 2 weighted MLE",
		family: NewFamily(PoissonFamily),
		start:  nil,
		data:   data2(true),
		weight: true,
		params: []float64{-1.540684, 0.116108, 0.246615},
		stderr: []float64{0.775912, 0.135982, 0.283345},
		vcov: []float64{0.602039, -0.076174, -0.174483,
			-0.076174, 0.018491, 0.019897,
			-0.174483, 0.019897, 0.080284},
		ll:         -13.098177137990557,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Poisson 3 weighted MLE",
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data3(true),
		weight:     true,
		params:     []float64{-0.896361, 0.467334},
		stderr:     []float64{0.428867, 0.647330},
		vcov:       []float64{0.183927, -0.157139, -0.157139, 0.419036},
		ll:         -13.768882387425702,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Binomial 1 weighted MLE",
		family: NewFamily(BinomialFamily),
		start:  nil,
		data:   data2(true),
		weight: true,
		params: []float64{-1.378328, 0.201911, 0.407917},
		stderr: []float64{0.927975, 0.187708, 0.363425},
		vcov: []float64{0.861138, -0.122218, -0.258570, -0.122218, 0.035234, 0.037427,
			-0.258570, 0.037427, 0.132078},
		ll:         -11.17418536789415,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Binomial 2 weighted MLE",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data3(true),
		weight:     true,
		params:     []float64{-0.343610, 0.934519},
		stderr:     []float64{0.553523, 0.963054},
		vcov:       []float64{0.306388, -0.227123, -0.227123, 0.927473},
		ll:         -11.245509472906111,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Binomial 3 unweighted MLE",
		family: NewFamily(BinomialFamily),
		start:  nil,
		data:   data2(false),
		params: []float64{-1.650145, 0.190136, 0.344331},
		stderr: []float64{1.505798, 0.323601, 0.593428},
		vcov: []float64{2.267429, -0.337163, -0.684836,
			-0.337163, 0.104718, 0.116028,
			-0.684836, 0.116028, 0.352157},
		ll:         -3.9607532681097091,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Binomial 4 unweighted MLE",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data3(false),
		params:     []float64{-0.434175, 0.868350},
		stderr:     []float64{0.830041, 1.306904},
		vcov:       []float64{0.688967, -0.330063, -0.330063, 1.707998},
		ll:         -4.53963553741,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Poisson 4 unweighted MLE",
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data1(false),
		params:     []float64{0.213361, -0.081530},
		stderr:     []float64{0.357095, 0.100337},
		vcov:       []float64{0.127517, -0.005034, -0.005034, 0.010067},
		ll:         -9.1041354864426385,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Poisson 5 unweighted MLE",
		family: NewFamily(PoissonFamily),
		start:  nil,
		data:   data2(false),
		params: []float64{-1.792499, 0.128696, 0.241203},
		stderr: []float64{1.325076, 0.256408, 0.496363},
		vcov: []float64{1.755827, -0.241115, -0.515732,
			-0.241115, 0.065745, 0.073083,
			-0.515732, 0.073083, 0.246377},
		ll:         -4.3466061504389559,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Poisson 6 unweighted MLE",
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data3(false),
		params:     []float64{-0.962424, 0.481212},
		stderr:     []float64{0.656431, 0.937078},
		vcov:       []float64{0.430902, -0.292705, -0.292705, 0.878115},
		ll:         -5.4060591253,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Gaussian 4 unweighted MLE",
		family:     NewFamily(GaussianFamily),
		start:      nil,
		data:       data1(false),
		params:     []float64{1.290837, -0.103586},
		stderr:     []float64{0.456706, 0.130298},
		vcov:       []float64{0.208581, -0.024254, -0.024254, 0.016978},
		ll:         -9.621454,
		scale:      1.21752988048,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Gaussian 5",
		family: NewFamily(GaussianFamily),
		start:  nil,
		data:   data2(false),
		params: []float64{0.154198, 0.038670, 0.066739},
		stderr: []float64{0.333030, 0.083695, 0.142159},
		vcov: []float64{0.110909, -0.017874, -0.032931,
			-0.017874, 0.007005, 0.006884,
			-0.032931, 0.006884, 0.020209},
		ll:         -4.596270,
		scale:      0.334176605228,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Gaussian 6",
		family:     NewFamily(GaussianFamily),
		start:      nil,
		data:       data3(false),
		params:     []float64{0.4, 0.2},
		stderr:     []float64{0.219089, 0.334664},
		vcov:       []float64{0.048, -0.016, -0.016, 0.112},
		ll:         -4.944550,
		scale:      0.32,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Inverse Gaussian 1",
		family: NewFamily(InvGaussianFamily),
		start:  []float64{0.1, 0, 0},
		data:   data4(true),
		weight: true,
		params: []float64{0.091657, -0.001893, -0.000376},
		stderr: []float64{0.033456, 0.009052, 0.014820},
		vcov: []float64{0.001119, -0.000203, -0.000353,
			-0.000203, 0.000082, 0.000086,
			-0.000353, 0.000086, 0.000220},
		ll:         -33.701849656107399,
		scale:      0.074887605672913735,
		fitmethods: []string{"IRLS"},
	},
	{
		title:  "Gamma 1",
		family: NewFamily(GammaFamily),
		start:  []float64{0.3, 0.0, 0.0},
		data:   data4(true),
		weight: true,
		params: []float64{0.302721, -0.003171, -0.000705},
		stderr: []float64{0.055975, 0.015255, 0.024878},
		vcov: []float64{0.003133, -0.000569, -0.000999,
			-0.000569, 0.000233, 0.000250,
			-0.000999, 0.000250, 0.000619},
		ll:         -31.687753839200358,
		scale:      0.25143442760931506,
		fitmethods: []string{"IRLS"},
	},
	{
		title:  "QuasiPoisson 1",
		family: NewFamily(QuasiPoissonFamily),
		start:  nil,
		data:   data2(true),
		weight: true,
		params: []float64{-1.540684, 0.116108, 0.246615},
		stderr: []float64{0.684396089274111, 0.11994309040228321, 0.24992565563491265},
		vcov: []float64{0.46839800701369694, -0.05926478926076194, -0.13575141649912711,
			-0.05926478926076193, 0.014386344935250282, 0.015480514629643507,
			-0.1357514164991272, 0.015480514629643507, 0.06246283334454094},
		ll:         -13.098177137990557,
		scale:      0.7780190501841399,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:  "Negative binomial 1",
		family: NewNegBinomFamily(1, NewLink(LogLink)),
		start:  nil,
		data:   data4(true),
		weight: true,
		params: []float64{1.191131, 0.011851, 0.004809},
		stderr: []float64{0.187331, 0.051733, 0.083739},
		vcov: []float64{0.035093, -0.006365, -0.011408,
			-0.006365, 0.002676, 0.002968,
			-0.011408, 0.002968, 0.007012},
		ll:         -39.875709730019153,
		scale:      0.19468567690459238,
		fitmethods: []string{"IRLS"}, // Gradient does not converge
	},
	{
		title:  "Negative binomial 2",
		family: NewNegBinomFamily(1.5, NewLink(LogLink)),
		start:  nil,
		data:   data4(true),
		weight: true,
		params: []float64{1.190715, 0.011981, 0.005043},
		stderr: []float64{0.187342, 0.051768, 0.083768},
		vcov: []float64{0.035097, -0.006366, -0.011417,
			-0.006366, 0.002680, 0.002974,
			-0.011417, 0.002974, 0.007017},
		ll:         -42.669972197288509,
		scale:      0.14064363313622641,
		fitmethods: []string{"IRLS"}, // Gradient does not converge
	},
	{
		title:      "Poisson 7",
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data5(true),
		weight:     true,
		offset:     true,
		params:     []float64{-0.183029, -0.075427},
		stderr:     []float64{0.236279, 0.074241},
		vcov:       []float64{0.055828, -0.001225, -0.001225, 0.005512},
		ll:         -15.259195632772048,
		scale:      1.0,
		fitmethods: []string{"Gradient", "IRLS"},
	},
	{
		title:      "Poisson 8",
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data1(true),
		weight:     true,
		params:     []float64{0.256717, -0.035340},
		scale:      1.0,
		l2wgt:      map[string]float64{"x1": 0.1, "x2": 0.1},
		fitmethods: []string{"Gradient"},
	},
	{
		title:      "Poisson 9",
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data2(true),
		weight:     true,
		params:     []float64{-0.921685, 0.032864, 0.064429},
		scale:      1.0,
		l2wgt:      map[string]float64{"x1": 0.2, "x2": 0.2, "x3": 0.2},
		fitmethods: []string{"Gradient"},
	},
	{
		title:      "Binomial 6",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(true),
		weight:     true,
		params:     []float64{-0.640768, 0.092631, 0.175485},
		scale:      1.0,
		l2wgt:      map[string]float64{"x1": 0.2, "x2": 0.2, "x3": 0.2},
		fitmethods: []string{"Gradient"},
	},
	{
		title:      "Binomial 7",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(true),
		weight:     true,
		params:     []float64{-0.659042, 0.097647, 0.187009},
		scale:      1.0,
		l2wgt:      map[string]float64{"x1": 0.2, "x2": 0, "x3": 0.1},
		fitmethods: []string{"Gradient"},
	},
	{
		title:      "Binomial 8 L1-regularized",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-0.465363, 0, 0},
		scale:      1.0,
		l1wgt:      map[string]float64{"x1": 0.1, "x2": 0.1, "x3": 0.1},
		fitmethods: []string{"Coordinate"},
	},
	{
		title:      "Binomial 9 L1-regularized",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-0.737198, 0.024176, 0.017089},
		scale:      1.0,
		l1wgt:      map[string]float64{"x1": 0.05, "x2": 0.05, "x3": 0.05},
		fitmethods: []string{"Coordinate"},
	},
	{
		title:      "Binomial 10 L1-regularized",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-1.433479, 0.152795, 0.268036},
		scale:      1.0,
		l1wgt:      map[string]float64{"x1": 0.01, "x2": 0.01, "x3": 0.01},
		fitmethods: []string{"Coordinate"},
	},
	{
		title:      "Binomial 11 L1 and L2 regularized",
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-0.988257, 0.078329, 0.121922},
		scale:      1.0,
		l1wgt:      map[string]float64{"x1": 0.02, "x2": 0.02, "x3": 0.02},
		l2wgt:      map[string]float64{"x1": 0.02, "x2": 0.02, "x3": 0.02},
		fitmethods: []string{"Coordinate"},
	},
	{
		title:      "Tweedie 1",
		family:     NewTweedieFamily(1.5, NewLink(LogLink)),
		start:      nil,
		data:       data1(false),
		weight:     false,
		params:     []float64{0.22297879, -0.09520094},
		stderr:     []float64{0.3885426, 0.1102231},
		ll:         -11.33792,
		scale:      1.043098,
		vcov:       []float64{0.15096534, -0.01102984, -0.01102984, 0.01214913},
		l1wgt:      nil,
		l2wgt:      nil,
		fitmethods: []string{"IRLS", "Coordinate"},
	},
	{
		title:  "Tweedie 2",
		family: NewTweedieFamily(1.75, NewLink(LogLink)),
		start:  nil,
		data:   data2(false),
		weight: false,
		params: []float64{-1.78192221, 0.05615625, 0.31635418},
		stderr: []float64{1.3368212, 0.3106755, 0.5459943},
		ll:     -8.484231,
		scale:  3.465428,
		vcov: []float64{1.7870908, -0.27012970, -0.5472654, -0.2701297, 0.09651928, 0.1062641,
			-0.5472654, 0.10626409, 0.2981098},
		vcovtol:    1e-3,
		paramstol:  1e-3,
		stderrtol:  1e-3,
		l1wgt:      nil,
		l2wgt:      nil,
		fitmethods: []string{"IRLS", "Coordinate"},
	},
}

func TestFit(t *testing.T) {

	lf, err := os.Create("glm_test.log")
	if err != nil {
		panic(err)
	}
	defer lf.Close()
	tlog := log.New(lf, "", log.Lshortfile)

	for _, ds := range glmTests {
		for _, fmeth := range ds.fitmethods {

			config := DefaultConfig()
			config.Family = ds.family
			config.FitMethod = fmeth

			if ds.weight {
				config.WeightVar = "w"
			}

			if ds.offset {
				config.OffsetVar = "off"
			}

			if ds.l1wgt != nil {
				config.L1Penalty = ds.l1wgt
			}

			if ds.l2wgt != nil {
				config.L2Penalty = ds.l2wgt
			}

			if ds.start != nil {
				config.Start = ds.start
			}

			lf, err := os.Create("glm.log")
			if err != nil {
				panic(err)
			}
			defer lf.Close()
			config.Log = log.New(lf, "", log.Lshortfile)

			glm := NewGLM(ds.data, config)
			result := glm.Fit()

			if ds.paramstol == 0 {
				ds.paramstol = 1e-5
			}

			if ds.stderrtol == 0 {
				ds.stderrtol = 1e-5
			}

			if ds.vcovtol == 0 {
				ds.vcovtol = 1e-5
			}

			if ds.logliketol == 0 {
				ds.logliketol = 1e-5
			}

			if ds.scaletol == 0 {
				ds.scaletol = 1e-5
			}

			if !floats.EqualApprox(result.Params(), ds.params, ds.paramstol) {
				tlog.Printf("%s\n", ds.title)
				tlog.Printf("Model: %+v\n", glm)
				tlog.Printf("Problem: %+v\n", ds)
				tlog.Printf("Parameter estimates disagree:\n")
				tlog.Printf("Expected: %v\n", ds.params)
				tlog.Printf("Found:    %v\n", result.Params())
				t.Fail()
			}

			if math.Abs(result.Scale()-ds.scale) > ds.scaletol {
				tlog.Printf("%s\n", ds.title)
				tlog.Printf("Model: %+v\n", glm)
				tlog.Printf("Problem: %+v\n", ds)
				tlog.Printf("Scale estimates disagree:\n")
				tlog.Printf("Expected: %v\n", ds.scale)
				tlog.Printf("Found:    %v\n", result.Scale())
				t.Fail()
			}

			// No stderr or vcov with regularization
			if ds.l2wgt != nil || ds.l1wgt != nil {
				continue
			}

			if !scalarClose(result.LogLike(), ds.ll, ds.logliketol) {
				tlog.Printf("%s\n", ds.title)
				tlog.Printf("Model: %+v\n", glm)
				tlog.Printf("Problem: %+v\n", ds)
				tlog.Printf("Loglikelihood values disagree:\n")
				tlog.Printf("Expected: %v\n", ds.ll)
				tlog.Printf("Found:    %v\n", result.LogLike())
				t.Fail()
			}

			if !floats.EqualApprox(result.StdErr(), ds.stderr, ds.stderrtol) {
				tlog.Printf("%s\n", ds.title)
				tlog.Printf("Model: %+v\n", glm)
				tlog.Printf("Problem: %+v\n", ds)
				tlog.Printf("Standard errors disagree\n")
				tlog.Printf("Expected: %v\n", ds.stderr)
				tlog.Printf("Found:    %v\n", result.StdErr())
				t.Fail()
			}

			if !floats.EqualApprox(result.VCov(), ds.vcov, ds.vcovtol) {
				tlog.Printf("%s\n", ds.title)
				tlog.Printf("Model: %+v\n", glm)
				tlog.Printf("Problem: %+v\n", ds)
				tlog.Printf("vcov values disagree\n")
				tlog.Printf("Expected: %v\n", ds.vcov)
				tlog.Printf("Found:    %v\n", result.VCov())
				t.Fail()
			}

			// Smoke test
			_ = result.Summary()
		}
	}
}

func TestSetLink(t *testing.T) {

	fam := NewFamily(BinomialFamily)
	for _, v := range []LinkType{LogitLink, LogLink, IdentityLink} {
		if !fam.IsValidLink(NewLink(v)) {
			t.Fail()
		}
	}
}
