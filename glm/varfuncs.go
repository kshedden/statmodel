package glm

import (
	"fmt"
)

// VarianceType is sed to specify a GLM variance function.
type VarianceType uint8

const (
	BinomialVar VarianceType = iota
	IdentityVar
	ConstantVar
	SquaredVar
	CubedVar
)

// NewVariance returns a new variance function object corresponding to
// the given name.  Supported names are binomial, const, cubed,
// identity, and, squared.
func NewVariance(vartype VarianceType) *Variance {

	switch vartype {
	case BinomialVar:
		return &binomVariance
	case IdentityVar:
		return &identVariance
	case ConstantVar:
		return &constVariance
	case SquaredVar:
		return &squaredVariance
	case CubedVar:
		return &cubedVariance
	default:
		msg := fmt.Sprintf("Unknown variance function: %d\n", vartype)
		panic(msg)
	}
}

// Variance represents a GLM variance function.
type Variance struct {
	Name  string
	Var   VecFunc
	Deriv VecFunc
}

var binomVariance = Variance{
	Name:  "Binomial",
	Var:   binomVar,
	Deriv: binomVarDeriv,
}

var identVariance = Variance{
	Name:  "Identity",
	Var:   identVar,
	Deriv: identVarDeriv,
}

var constVariance = Variance{
	Name:  "Constant",
	Var:   constVar,
	Deriv: constVarDeriv,
}

var squaredVariance = Variance{
	Name:  "Squared",
	Var:   squaredVar,
	Deriv: squaredVarDeriv,
}

var cubedVariance = Variance{
	Name:  "Cubed",
	Var:   cubedVar,
	Deriv: cubedVarDeriv,
}

func binomVar(mn []float64, v []float64) {
	for i, p := range mn {
		v[i] = p * (1 - p)
	}
}

func binomVarDeriv(mn []float64, dv []float64) {
	for i, p := range mn {
		dv[i] = 1 - 2*p
	}
}

func identVar(mn []float64, v []float64) {
	copy(v, mn)
}

func identVarDeriv(mn []float64, v []float64) {
	one(v)
}

func constVar(mn []float64, v []float64) {
	one(v)
}

func constVarDeriv(mn []float64, v []float64) {
	zero(v)
}

func squaredVar(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = m * m
	}
}

func squaredVarDeriv(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = 2 * m
	}
}

func cubedVar(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = m * m * m
	}
}

func cubedVarDeriv(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = 3 * m * m
	}
}

// NewNegBinomVariance returns a variance function for the negative
// binomial family, using the given parameter alpha to determine the
// mean/variance relationship.  The variance for mean m is m +
// alpha*m^2.
func NewNegBinomVariance(alpha float64) *Variance {

	vaf := func(mn []float64, v []float64) {
		for i, m := range mn {
			v[i] = m + alpha*m*m
		}
	}

	vad := func(mn []float64, v []float64) {
		for i, m := range mn {
			v[i] = 1 + 2*alpha*m
		}
	}

	return &Variance{
		Var:   vaf,
		Deriv: vad,
	}
}
