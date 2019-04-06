package glm

import (
	"fmt"
	"math"
	"testing"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/gonum/floats"
)

func scalarClose(x, y, eps float64) bool {
	if math.Abs(x-y) > eps {
		return false
	}
	return true
}

func data1(wgt bool) dstream.Dstream {

	y := []float64{0, 1, 3, 2, 1, 1, 0}
	x1 := []float64{1, 1, 1, 1, 1, 1, 1}
	x2 := []float64{4, 1, -1, 3, 5, -5, 3}
	w := []float64{1, 2, 2, 3, 1, 3, 2}
	da := []interface{}{y, x1, x2}
	na := []string{"y", "x1", "x2"}

	if wgt {
		da = append(da, w)
		na = append(na, "w")
	}

	return dstream.NewFromFlat(da, na)
}

func data2(wgt bool) dstream.Dstream {

	y := []float64{0, 0, 1, 0, 1, 0, 0}
	x1 := []float64{1, 1, 1, 1, 1, 1, 1}
	x2 := []float64{4, 1, -1, 3, 5, -5, 3}
	x3 := []float64{1, -1, 1, 1, 2, 5, -1}
	w := []float64{2, 1, 3, 3, 4, 2, 3}

	da := []interface{}{y, x1, x2, x3}
	na := []string{"y", "x1", "x2", "x3"}

	if wgt {
		da = append(da, w)
		na = append(na, "w")
	}

	return dstream.NewFromFlat(da, na)
}

func data3(wgt bool) dstream.Dstream {

	y := []float64{1, 1, 1, 0, 0, 0, 0}
	x1 := []float64{1, 1, 1, 1, 1, 1, 1}
	x2 := []float64{0, 1, 0, 0, -1, 0, 1}
	w := []float64{3, 3, 2, 3, 1, 3, 2}

	da := []interface{}{y, x1, x2}
	na := []string{"y", "x1", "x2"}

	if wgt {
		da = append(da, w)
		na = append(na, "w")
	}

	return dstream.NewFromFlat(da, na)
}

func data4(wgt bool) dstream.Dstream {

	y := []float64{3, 1, 5, 4, 2, 3, 6}
	x1 := []float64{1, 1, 1, 1, 1, 1, 1}
	x2 := []float64{4, 1, -1, 3, 5, -5, 3}
	x3 := []float64{1, -1, 1, 1, 2, 5, -1}
	w := []float64{3, 3, 2, 3, 1, 3, 2}

	da := []interface{}{y, x1, x2, x3}
	na := []string{"y", "x1", "x2", "x3"}

	if wgt {
		da = append(da, w)
		na = append(na, "w")
	}

	return dstream.NewFromFlat(da, na)
}

func data5(wgt bool) dstream.Dstream {

	y := []float64{0, 1, 3, 2, 1, 1, 0}
	x1 := []float64{1, 1, 1, 1, 1, 1, 1}
	x2 := []float64{4, 1, -1, 3, 5, -5, 3}
	off := []float64{0, 0, 1, 1, 0, 0, 0}
	w := []float64{1, 2, 2, 3, 1, 3, 2}

	da := []interface{}{y, x1, x2, off}
	na := []string{"y", "x1", "x2", "off"}

	if wgt {
		da = append(da, w)
		na = append(na, "w")
	}

	return dstream.NewFromFlat(da, na)
}

// A test problem
type testprob struct {
	family     *Family
	data       dstream.Dstream
	weight     bool
	offset     bool
	alpha      float64
	start      []float64
	params     []float64
	stderr     []float64
	vcov       []float64
	ll         float64
	scale      float64
	l2wgt      []float64
	l1wgt      []float64
	fitmethods []string
	scaletype  []statmodel.ScaleType
}

var glm_tests []testprob = []testprob{
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data3(false),
		params:     []float64{-0.434175, 0.868350},
		stderr:     []float64{0.830041, 1.306904},
		vcov:       []float64{0.688967, -0.330063, -0.330063, 1.707998},
		ll:         -4.53963553741,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data1(false),
		params:     []float64{0.213361, -0.081530},
		stderr:     []float64{0.357095, 0.100337},
		vcov:       []float64{0.127517, -0.005034, -0.005034, 0.010067},
		ll:         -9.1041354864426385,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data3(false),
		params:     []float64{-0.962424, 0.481212},
		stderr:     []float64{0.656431, 0.937078},
		vcov:       []float64{0.430902, -0.292705, -0.292705, 0.878115},
		ll:         -5.4060591253,
		scale:      1,
		fitmethods: []string{"Gradient", "IRLS"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		family:     NewFamily(GaussianFamily),
		start:      nil,
		data:       data1(false),
		params:     []float64{1.290837, -0.103586},
		stderr:     []float64{0.456706, 0.130298},
		vcov:       []float64{0.208581, -0.024254, -0.024254, 0.016978},
		ll:         -9.621454,
		scale:      1.21752988048,
		fitmethods: []string{"Gradient", "IRLS"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
		family:     NewFamily(GaussianFamily),
		start:      nil,
		data:       data3(false),
		params:     []float64{0.4, 0.2},
		stderr:     []float64{0.219089, 0.334664},
		vcov:       []float64{0.048, -0.016, -0.016, 0.112},
		ll:         -4.944550,
		scale:      0.32,
		fitmethods: []string{"Gradient", "IRLS"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm, statmodel.Variance},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm, statmodel.Variance},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm, statmodel.Variance},
	},
	{
		family: NewNegBinomFamily(1.5, NewLink(LogLink)),
		alpha:  1.5,
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm, statmodel.Variance},
	},
	{
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
		scaletype:  []statmodel.ScaleType{statmodel.NoScale, statmodel.L2Norm, statmodel.Variance},
	},
	{
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data1(true),
		weight:     true,
		params:     []float64{0.256717, -0.035340},
		scale:      1.0,
		l2wgt:      []float64{0.1, 0.1},
		fitmethods: []string{"Gradient"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale},
	},
	{
		family:     NewFamily(PoissonFamily),
		start:      nil,
		data:       data2(true),
		weight:     true,
		params:     []float64{-0.921685, 0.032864, 0.064429},
		scale:      1.0,
		l2wgt:      []float64{0.2, 0.2, 0.2},
		fitmethods: []string{"Gradient"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale},
	},
	{
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(true),
		weight:     true,
		params:     []float64{-0.640768, 0.092631, 0.175485},
		scale:      1.0,
		l2wgt:      []float64{0.2, 0.2, 0.2},
		fitmethods: []string{"Gradient"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale},
	},
	{
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(true),
		weight:     true,
		params:     []float64{-0.659042, 0.097647, 0.187009},
		scale:      1.0,
		l2wgt:      []float64{0.2, 0, 0.1},
		fitmethods: []string{"Gradient"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale},
	},
	{
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-0.465363, 0, 0},
		scale:      1.0,
		l1wgt:      []float64{0.1, 0.1, 0.1},
		fitmethods: []string{"Coordinate"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale},
	},
	{
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-0.737198, 0.024176, 0.017089},
		scale:      1.0,
		l1wgt:      []float64{0.05, 0.05, 0.05},
		fitmethods: []string{"Coordinate"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale},
	},
	{
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-0.89149010, 0.0312166489, 0.0485293176},
		scale:      1.0,
		l1wgt:      []float64{0.01, 0.01, 0.01},
		fitmethods: []string{"Coordinate"},
		scaletype:  []statmodel.ScaleType{statmodel.L2Norm},
	},
	{
		family:     NewFamily(BinomialFamily),
		start:      nil,
		data:       data2(false),
		weight:     false,
		params:     []float64{-0.988257, 0.078329, 0.121922},
		scale:      1.0,
		l1wgt:      []float64{0.02, 0.02, 0.02},
		l2wgt:      []float64{0.02, 0.02, 0.02},
		fitmethods: []string{"Coordinate"},
		scaletype:  []statmodel.ScaleType{statmodel.NoScale},
	},
}

func TestFit(t *testing.T) {

	for jd, ds := range glm_tests {
		for js, scaletype := range ds.scaletype {
			for jf, fmeth := range ds.fitmethods {

				var glm *GLM
				glm = NewGLM(ds.data, "y")

				if ds.weight {
					glm = glm.Weight("w")
				}

				if ds.offset {
					glm = glm.Offset("off")
				}

				glm = glm.Family(ds.family).FitMethod(fmeth)

				if ds.l2wgt != nil {
					glm = glm.L2Weight(ds.l2wgt)
				}

				glm = glm.CovariateScale(scaletype)

				if len(ds.start) > 0 {
					glm = glm.Start(ds.start)
				}

				if len(ds.l1wgt) > 0 {
					glm = glm.L1Weight(ds.l1wgt)
				}

				glm = glm.Done()
				result := glm.Fit()

				if !floats.EqualApprox(result.Params(), ds.params, 1e-5) {
					fmt.Printf("params failed %d %d %d:\n", jd, js, jf)
					fmt.Printf("%v\n", result.Params())
					t.Fail()
				}

				if math.Abs(result.Scale()-ds.scale) > 1e-5 {
					fmt.Printf("scale failed: %d %d %d\n", jd, js, jf)
					t.Fail()
				}

				// No stderr or vcov with regularization
				if ds.l2wgt != nil || ds.l1wgt != nil {
					continue
				}

				if !scalarClose(result.LogLike(), ds.ll, 1e-5) {
					fmt.Printf("loglike failed: %d %d %d\n", jd, js, jf)
					t.Fail()
				}

				if !floats.EqualApprox(result.StdErr(), ds.stderr, 1e-5) {
					fmt.Printf("stderr failed: %d %d %d\n", jd, js, jf)
					t.Fail()
				}

				if !floats.EqualApprox(result.VCov(), ds.vcov, 1e-5) {
					fmt.Printf("vcov failed: %d %d %d\n", jd, js, jf)
					t.Fail()
				}

				// Smoke test
				_ = result.Summary()
			}
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
