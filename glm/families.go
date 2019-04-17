package glm

import (
	"fmt"
	"math"
)

// FamilyType is the type of GLM family used in a model.
type FamilyType uint8

// BinomialFamily, etc. are the GLM family types.
const (
	BinomialFamily FamilyType = iota
	PoissonFamily
	QuasiPoissonFamily
	GaussianFamily
	GammaFamily
	InvGaussianFamily
	NegBinomFamily
)

// LogLikeFunc evaluates and returns the log-likelihood for a GLM.  The arguments
// are the data, the mean values, the weights, and the scale parameter.  The weights
// may be nil in which case all weights are taken to be 1.
type LogLikeFunc func([]float64, []float64, []float64, float64) float64

// DevianceFunc evaluates and returns the deviance for a GLM.  The arguments
// are the data, the mean values, the weights, and the scale parameter.  The weights
// may be nil in which case all weights are taken to be 1.
type DevianceFunc func([]float64, []float64, []float64, float64) float64

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

	// The names of valid links for this family.  The first listed
	// link should be the canonical link.
	validLinks []LinkType

	// The link in use by the family, only specified for negative binomial
	link *Link

	// Negatie binomial parameter
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
	Name:       "Poisson",
	TypeCode:   PoissonFamily,
	LogLike:    poissonLogLike,
	Deviance:   poissonDeviance,
	validLinks: []LinkType{LogLink, IdentityLink},
}

// QuasiPoisson is the same as Poisson, except that the scale parameter is estimated.
var quasiPoisson = Family{
	Name:       "QuasiPoisson",
	TypeCode:   QuasiPoissonFamily,
	LogLike:    poissonLogLike,
	Deviance:   poissonDeviance,
	validLinks: []LinkType{LogLink, IdentityLink},
}

var binomial = Family{
	Name:       "Binomial",
	TypeCode:   BinomialFamily,
	LogLike:    binomialLogLike,
	Deviance:   binomialDeviance,
	validLinks: []LinkType{LogitLink, LogLink, IdentityLink},
}

var gaussian = Family{
	Name:       "Gaussian",
	TypeCode:   GaussianFamily,
	LogLike:    gaussianLogLike,
	Deviance:   gaussianDeviance,
	validLinks: []LinkType{IdentityLink, LogLink, RecipLink},
}

var gamma = Family{
	Name:       "Gamma",
	TypeCode:   GammaFamily,
	LogLike:    gammaLogLike,
	Deviance:   gammaDeviance,
	validLinks: []LinkType{RecipLink, LogLink, IdentityLink},
}

var invGaussian = Family{
	Name:       "InvGaussian",
	TypeCode:   InvGaussianFamily,
	LogLike:    invGaussLogLike,
	Deviance:   invGaussianDeviance,
	validLinks: []LinkType{RecipSquaredLink, RecipLink, LogLink, IdentityLink},
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

func poissonLogLike(y, mn, wt []float64, scale float64) float64 {
	var ll float64
	var w float64 = 1
	for i := 0; i < len(y); i++ {
		g, _ := math.Lgamma(y[i] + 1)
		if wt != nil {
			w = wt[i]
		}
		ll += w * (y[i]*math.Log(mn[i]) - mn[i] - g)
	}
	return ll
}

func binomialLogLike(y, mn, wt []float64, scale float64) float64 {
	var ll float64
	var w float64 = 1
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}
		r := mn[i]/(1-mn[i]) + 1e-200
		ll += w * (y[i]*math.Log(r) + math.Log(1-mn[i]))
	}
	return ll
}

func gaussianLogLike(y, mn, wt []float64, scale float64) float64 {
	var ll float64
	var w float64 = 1
	var ws float64
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}
		r := y[i] - mn[i]
		ll -= w * r * r / (2 * scale)
		ws += w
	}
	ll -= ws * math.Log(2*math.Pi*scale) / 2
	return ll
}

func gammaLogLike(y, mn, wt []float64, scale float64) float64 {

	var ll float64
	var w float64 = 1
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}

		g, _ := math.Lgamma(1 / scale)
		v := y[i]/mn[i] + math.Log(mn[i]) + (scale-1)*math.Log(y[i])
		v += math.Log(scale) + scale*g
		ll -= w * v / scale
	}

	return ll
}

func invGaussLogLike(y, mn, wt []float64, scale float64) float64 {

	var ll float64
	var w float64 = 1
	var ws float64
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}

		r := y[i] - mn[i]
		v := r * r / (y[i] * mn[i] * mn[i] * scale)
		v += math.Log(scale * y[i] * y[i] * y[i])

		ll -= 0.5 * w * v
		ws += w
	}

	ll -= 0.5 * ws * math.Log(2*math.Pi)
	return ll
}

func poissonDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		if y[i] > 0 {
			dev += 2 * w * y[i] * math.Log(y[i]/mn[i])
		}
	}
	dev /= scale

	return dev
}

func binomialDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		dev -= 2 * w * (y[i]*math.Log(mn[i]) + (1-y[i])*math.Log(1-mn[i]))
	}

	return dev
}

func gammaDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		dev += 2 * w * ((y[i]-mn[i])/mn[i] - math.Log(y[i]/mn[i]))
	}

	return dev
}

func invGaussianDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		r := y[i] - mn[i]
		dev += w * (r * r / (y[i] * mn[i] * mn[i]))
	}
	dev /= scale

	return dev
}

func gaussianDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		r := y[i] - mn[i]
		dev += w * r * r
	}
	dev /= scale

	return dev
}

// NewNegBinomFamily returns a new family object for the negative
// binomial family, using the given link function.
func NewNegBinomFamily(alpha float64, link *Link) *Family {

	loglike := func(y, mn, wt []float64, scale float64) float64 {

		var ll float64
		var w float64 = 1
		var lp []float64

		lp = resize(lp, len(y))
		link.Link(mn, lp)
		c3, _ := math.Lgamma(1 / alpha)

		for i := 0; i < len(y); i++ {
			if wt != nil {
				w = wt[i]
			}

			elp := math.Exp(lp[i])

			c1, _ := math.Lgamma(y[i] + 1/alpha)
			c2, _ := math.Lgamma(y[i] + 1)
			c := c1 - c2 - c3

			v := y[i] * math.Log(alpha*elp/(1+alpha*elp))
			v -= math.Log(1+alpha*elp) / alpha

			ll += w * (v + c)
		}

		return ll
	}

	deviance := func(y, mn, wt []float64, scale float64) float64 {

		var dev float64
		var w float64 = 1
		var lp []float64

		lp = resize(lp, len(y))
		link.Link(mn, lp)

		for i := 0; i < len(y); i++ {
			if wt != nil {
				w = wt[i]
			}

			if y[i] == 1 {
				z1 := y[i] * math.Log(y[i]/mn[i])
				z2 := (1 + alpha*y[i]) / alpha
				z2 *= math.Log((1 + alpha*y[i]) / (1 + alpha*mn[i]))
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
	}
}
