package glm

import (
	"fmt"
	"math"

	"github.com/kshedden/statmodel/statmodel"
)

// FamilyType is the type of GLM family used in a model.
type FamilyType uint8

// BinomialFamily, ... are families for a GLM.
const (
	BinomialFamily FamilyType = iota
	PoissonFamily
	QuasiPoissonFamily
	GaussianFamily
	GammaFamily
	InvGaussianFamily
	NegBinomFamily
	TweedieFamily
)

// LogLikeFunc evaluates and returns the log-likelihood for a GLM.  The arguments
// are the data, the mean values, the weights, the scale parameter, and the 'exact flag'.
// If the exact flag is false, multiplicative factors that are constant with respect to
// the mean may be omitted.  The weights may be nil in which case all weights are taken to be 1.
type LogLikeFunc func([]statmodel.Dtype, []float64, []statmodel.Dtype, float64, bool) float64

// DevianceFunc evaluates and returns the deviance for a GLM.  The arguments
// are the data, the mean values, the weights, and the scale parameter.  The weights
// may be nil in which case all weights are taken to be 1.
type DevianceFunc func([]statmodel.Dtype, []float64, []statmodel.Dtype, float64) float64

// Family represents a generalized linear model family.
type Family struct {

	// The name of the family
	Name string

	// The numeric code for the family
	TypeCode FamilyType

	// The log-likelihood function for the family
	LogLike LogLikeFunc

	// The deviance function for the family
	Deviance DevianceFunc

	// The default approach for handling the dispersion if not set explicitly.
	dispersionDefaultMethod DispersionForm

	// The default dispersion value for a fixed dispersion
	dispersionDefaultValue float64

	// The names of valid links for this family.  The first listed
	// link should be the canonical link.
	validLinks []LinkType

	// The link in use by the family, only specified for negative binomial
	// and Tweedie
	// TODO cleanup
	link *Link

	// Auxiliary parameter: negative binomial parameter or Tweedie variance
	// power parameter
	alpha float64
}

// NewFamily returns a family object corresponding to the given name.
// Supported names are binomial, gamma, gaussian, invgaussian,
// poisson, quasipoisson.
func NewFamily(fam FamilyType) *Family {

	switch fam {
	case PoissonFamily:
		return &poisson
	case QuasiPoissonFamily:
		return &quasiPoisson
	case BinomialFamily:
		return &binomial
	case GaussianFamily:
		return &gaussian
	case GammaFamily:
		return &gamma
	case InvGaussianFamily:
		return &invGaussian
	default:
		msg := fmt.Sprintf("Unknown family: %v\n", fam)
		panic(msg)
	}
}

var poisson = Family{
	Name:                    "Poisson",
	TypeCode:                PoissonFamily,
	LogLike:                 poissonLogLike,
	Deviance:                poissonDeviance,
	validLinks:              []LinkType{LogLink, IdentityLink},
	dispersionDefaultMethod: DispersionFixed,
	dispersionDefaultValue:  1,
}

// QuasiPoisson is the same as Poisson, except that the scale parameter is estimated.
var quasiPoisson = Family{
	Name:                    "QuasiPoisson",
	TypeCode:                QuasiPoissonFamily,
	LogLike:                 poissonLogLike,
	Deviance:                poissonDeviance,
	validLinks:              []LinkType{LogLink, IdentityLink},
	dispersionDefaultMethod: DispersionFree,
	dispersionDefaultValue:  1,
}

var binomial = Family{
	Name:                    "Binomial",
	TypeCode:                BinomialFamily,
	LogLike:                 binomialLogLike,
	Deviance:                binomialDeviance,
	validLinks:              []LinkType{LogitLink, LogLink, IdentityLink},
	dispersionDefaultMethod: DispersionFixed,
	dispersionDefaultValue:  1,
}

var gaussian = Family{
	Name:                    "Gaussian",
	TypeCode:                GaussianFamily,
	LogLike:                 gaussianLogLike,
	Deviance:                gaussianDeviance,
	validLinks:              []LinkType{IdentityLink, LogLink, RecipLink},
	dispersionDefaultMethod: DispersionFree,
	dispersionDefaultValue:  1,
}

var gamma = Family{
	Name:                    "Gamma",
	TypeCode:                GammaFamily,
	LogLike:                 gammaLogLike,
	Deviance:                gammaDeviance,
	validLinks:              []LinkType{RecipLink, LogLink, IdentityLink},
	dispersionDefaultMethod: DispersionFree,
}

var invGaussian = Family{
	Name:                    "InvGaussian",
	TypeCode:                InvGaussianFamily,
	LogLike:                 invGaussLogLike,
	Deviance:                invGaussianDeviance,
	validLinks:              []LinkType{RecipSquaredLink, RecipLink, LogLink, IdentityLink},
	dispersionDefaultMethod: DispersionFree,
}

// IsValidLink returns true or false based on whether the link is
// valid for the family.
func (fam *Family) IsValidLink(link *Link) bool {

	for _, q := range fam.validLinks {
		if link.TypeCode == q {
			return true
		}
	}

	return false
}

func poissonLogLike(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64, exact bool) float64 {

	var ll float64
	var w float64 = 1
	for i := range y {
		if wt != nil {
			w = float64(wt[i])
		}
		ll += w * (float64(y[i])*math.Log(mn[i]) - mn[i])
	}

	if exact {
		for i := range y {
			if wt != nil {
				w = float64(wt[i])
			}
			g, _ := math.Lgamma(float64(y[i]) + 1)
			ll -= w * g
		}
	}

	return ll
}

func binomialLogLike(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64, exact bool) float64 {
	var ll float64
	var w float64 = 1
	for i := range y {
		if wt != nil {
			w = float64(wt[i])
		}
		r := mn[i]/(1-mn[i]) + 1e-200
		ll += w * (float64(y[i])*math.Log(r) + math.Log(1-mn[i]))
	}
	return ll
}

func gaussianLogLike(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64, exact bool) float64 {
	var ll float64
	var w float64 = 1
	var ws float64
	for i := range y {
		if wt != nil {
			w = float64(wt[i])
		}
		r := float64(y[i]) - mn[i]
		ll -= w * r * r / (2 * scale)
		ws += w
	}
	ll -= ws * math.Log(2*math.Pi*scale) / 2
	return ll
}

func gammaLogLike(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64, exact bool) float64 {

	var ll float64
	var w float64 = 1
	for i := range y {
		if wt != nil {
			w = float64(wt[i])
		}

		v := float64(y[i])/mn[i] + math.Log(mn[i])
		ll -= w * v / scale
	}

	if exact {
		for i := range y {
			if wt != nil {
				w = float64(wt[i])
			}

			v := (scale - 1) * math.Log(float64(y[i]))
			g, _ := math.Lgamma(1 / scale)
			v += math.Log(scale) + scale*g
			ll -= w * v / scale
		}
	}

	return ll
}

func invGaussLogLike(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64, exact bool) float64 {

	var ll float64
	var w float64 = 1
	var ws float64
	for i := range y {
		if wt != nil {
			w = float64(wt[i])
		}

		r := float64(y[i]) - mn[i]
		v := r * r / (float64(y[i]) * mn[i] * mn[i] * scale)

		ll -= 0.5 * w * v
		ws += w
	}
	ll -= 0.5 * ws * math.Log(2*math.Pi)

	if exact {
		for i := range y {
			if wt != nil {
				w = float64(wt[i])
			}
			ll -= 0.5 * w * math.Log(scale*float64(y[i]*y[i]*y[i]))
		}
	}

	return ll
}

func poissonDeviance(y []statmodel.Dtype, mn []float64, wgt []statmodel.Dtype, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := range y {
		if wgt != nil {
			w = float64(wgt[i])
		}

		if y[i] > 0 {
			dev += 2 * w * float64(y[i]) * math.Log(float64(y[i])/mn[i])
		}
	}
	dev /= scale

	return dev
}

func binomialDeviance(y []statmodel.Dtype, mn []float64, wgt []statmodel.Dtype, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := range y {
		if wgt != nil {
			w = float64(wgt[i])
		}

		dev -= 2 * w * (float64(y[i])*math.Log(mn[i]) + (1-float64(y[i]))*math.Log(1-mn[i]))
	}

	return dev
}

func gammaDeviance(y []statmodel.Dtype, mn []float64, wgt []statmodel.Dtype, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := range y {
		if wgt != nil {
			w = float64(wgt[i])
		}

		dev += 2 * w * ((float64(y[i])-mn[i])/mn[i] - math.Log(float64(y[i])/mn[i]))
	}

	return dev
}

func invGaussianDeviance(y []statmodel.Dtype, mn []float64, wgt []statmodel.Dtype, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := range y {
		if wgt != nil {
			w = float64(wgt[i])
		}

		r := float64(y[i]) - mn[i]
		dev += w * (r * r / (float64(y[i]) * mn[i] * mn[i]))
	}
	dev /= scale

	return dev
}

func gaussianDeviance(y []statmodel.Dtype, mn []float64, wgt []statmodel.Dtype, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := range y {
		if wgt != nil {
			w = float64(wgt[i])
		}

		r := float64(y[i]) - mn[i]
		dev += w * r * r
	}
	dev /= scale

	return dev
}

// NewTweedieFamily returns a new family object for the Tweedie
// family, using the given variance power and link function.  The
// variance power determines the mean/variance relationship,
// variance = mean^pw.  If link is nil, the canonical link is used,
// which is a power function with exponent 1 - pw.  Passing
// NewLink(LogLink) as the link gives the log link, which avoids
// domain violations.
func NewTweedieFamily(pw float64, link *Link) *Family {

	if link == nil {
		link = NewPowerLink(1 - pw)
	}

	loglike := func(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64, exact bool) float64 {
		var ll float64
		var w float64 = 1
		for i := range y {
			if wt != nil {
				w = float64(wt[i])
			}
			lmn := math.Log(mn[i])
			ll += w * (float64(y[i])*math.Exp((1-pw)*lmn)/(1-pw) - math.Exp((2-pw)*lmn)/(2-pw))
		}
		ll /= scale

		if exact {
			// calculate the Tweedie scaling factor here
			alp := float64(2-pw) / float64(1-pw)
			lscale := math.Log(scale)
			for i := range y {
				if wt != nil {
					w = float64(wt[i])
				}

				// Scaling factor is 1 in this case
				if y[i] == 0 {
					continue
				}

				lz := -alp*math.Log(float64(y[i])) + alp*math.Log(pw-1) - math.Log(2-pw) - (1-alp)*lscale
				kf := math.Pow(float64(y[i]), 2-pw) / (scale * float64(2-pw))
				k := int(math.Round(kf))
				if k < 1 {
					k = 1
				}

				// Sum the upper tail.
				w0 := float64(k)*lz - lgamma(float64(k+1)) - lgamma(-alp*float64(k))
				ws := 1.0
				for j := k + 1; j < 200; j++ {
					w1 := float64(j)*lz - lgamma(float64(j+1)) - lgamma(-alp*float64(j))
					if w1 < w0-37 {
						break
					}
					ws += math.Exp(w1 - w0)
					if j > 198 {
						println("Tweedie upper tail...")
					}
				}

				// Sum the lower tail.
				for j := k - 1; j > 0; j-- {
					w1 := float64(j)*lz - lgamma(float64(j+1)) - lgamma(-alp*float64(j))
					if w1 < w0-37 {
						break
					}
					ws += math.Exp(w1 - w0)
				}

				ll -= w * math.Log(float64(y[i]))
				ll += w * (w0 + math.Log(ws))
			}
		}

		return ll
	}

	deviance := func(y []statmodel.Dtype, mn []float64, wgt []statmodel.Dtype, scale float64) float64 {

		var dev float64
		var w float64 = 1

		for i := range y {
			if wgt != nil {
				w = float64(wgt[i])
			}

			u1 := math.Pow(float64(y[i]), 2-pw) / ((1 - pw) * (2 - pw))
			u2 := float64(y[i]) * math.Pow(mn[i], 1-pw) / (1 - pw)
			u3 := math.Pow(mn[i], 2-pw) / (2 - pw)
			dev += 2 * w * (u1 - u2 + u3)
		}
		dev /= scale

		return dev
	}

	return &Family{
		Name:       "Tweedie",
		TypeCode:   TweedieFamily,
		LogLike:    loglike,
		Deviance:   deviance,
		alpha:      pw,
		validLinks: []LinkType{LogLink, PowerLink},
		link:       link,
		dispersionDefaultMethod: DispersionFree,
	}
}

func lgamma(x float64) float64 {
	u, s := math.Lgamma(x)
	if s != 1 {
		panic("lgamma")
	}
	return u
}

// NewNegBinomFamily returns a new family object for the negative
// binomial family, using the given link function.
func NewNegBinomFamily(alpha float64, link *Link) *Family {

	loglike := func(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64, exact bool) float64 {

		var ll float64
		var w float64 = 1
		var lp []float64

		lp = resize(lp, len(y))
		link.Link(mn, lp)
		c3, _ := math.Lgamma(1 / alpha)

		for i := range y {

			if wt != nil {
				w = float64(wt[i])
			}

			elp := math.Exp(lp[i])

			c1, _ := math.Lgamma(float64(y[i]) + 1/alpha)
			c2, _ := math.Lgamma(float64(y[i]) + 1)
			c := c1 - c2 - c3

			v := float64(y[i]) * math.Log(alpha*elp/(1+alpha*elp))
			v -= math.Log(1+alpha*elp) / alpha

			ll += w * (v + c)
		}

		return ll
	}

	deviance := func(y []statmodel.Dtype, mn []float64, wt []statmodel.Dtype, scale float64) float64 {

		var dev float64
		var w float64 = 1
		var lp []float64

		lp = resize(lp, len(y))
		link.Link(mn, lp)

		for i := 0; i < len(y); i++ {
			if wt != nil {
				w = float64(wt[i])
			}

			if y[i] == 1 {
				z1 := float64(y[i]) * math.Log(float64(y[i])/mn[i])
				z2 := (1 + alpha*float64(y[i])) / alpha
				z2 *= math.Log((1 + alpha*float64(y[i])) / (1 + alpha*mn[i]))
				dev += w * (z1 - z2)
			} else {
				dev += 2 * w * math.Log(1+alpha*mn[i]) / alpha
			}
		}
		dev /= scale

		return dev
	}

	return &Family{
		Name:       "NegBinom",
		TypeCode:   NegBinomFamily,
		LogLike:    loglike,
		Deviance:   deviance,
		alpha:      alpha,
		validLinks: []LinkType{LogLink, IdentityLink},
		link:       link,
		dispersionDefaultMethod: DispersionFree,
	}
}
