package glm

import (
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/gonum/mat"
)

func (glm *GLM) fitIRLS(start []float64, maxiter int) []float64 {

	// TODO make this configurable
	dtol := 1e-8

	linpred := glm.getNslice()
	mn := glm.getNslice()
	va := glm.getNslice()
	lderiv := glm.getNslice()
	irlsw := glm.getNslice()
	adjy := glm.getNslice()

	var nparam mat.VecDense

	nvar := glm.NumParams()

	xty := make([]float64, nvar)
	xtx := make([]float64, nvar*nvar)

	var params []float64
	if start == nil {
		params = make([]float64, nvar)
	} else {
		params = start
	}

	var dev []float64

	xdat := make([][]statmodel.Dtype, len(glm.xpos))
	for j, k := range glm.xpos {
		xdat[j] = glm.data[k]
	}

	// IRLS iterations
	for iter := 0; iter < maxiter; iter++ {

		zero(xtx)
		zero(xty)
		var devi float64

		// Loop over data chunks
		var wgt, off []statmodel.Dtype

		yda := glm.data[glm.ypos]

		if glm.weightpos != -1 {
			wgt = glm.data[glm.weightpos]
		}
		if glm.offsetpos != -1 {
			off = glm.data[glm.offsetpos]
		}

		zero(linpred)
		for j := range glm.xpos {
			for i := range linpred {
				linpred[i] += float64(xdat[j][i]) * params[j]
			}
		}

		if off != nil {
			for i := range linpred {
				linpred[i] += float64(off[i])
			}
		}

		if iter == 0 {
			glm.startingMu(yda, mn)
		} else {
			glm.link.InvLink(linpred, mn)
		}

		glm.link.Deriv(mn, lderiv)
		glm.vari.Var(mn, va)

		devi += glm.fam.Deviance(yda, mn, wgt, 1)

		// Create weights for WLS
		if wgt != nil {
			for i := range yda {
				irlsw[i] = float64(wgt[i]) / (lderiv[i] * lderiv[i] * va[i])
			}
		} else {
			for i := range yda {
				irlsw[i] = 1 / (lderiv[i] * lderiv[i] * va[i])
			}
		}

		// Create an adjusted response for WLS
		if off == nil {
			for i := range yda {
				adjy[i] = linpred[i] + lderiv[i]*(float64(yda[i])-mn[i])
			}
		} else {
			for i := range yda {
				adjy[i] = linpred[i] + lderiv[i]*(float64(yda[i])-mn[i]) - float64(off[i])
			}
		}

		// Update the weighted moment matrices.  For large data sets, this is by far the
		// most expensive step.
		glm.irlsXprod(xdat, adjy, irlsw, xty, xtx)

		// Fill in the unfilled triangle of xtx
		for j1 := range glm.xpos {
			for j2 := j1 + 1; j2 < nvar; j2++ {
				xtx[j1*nvar+j2] = xtx[j2*nvar+j1]
			}
		}

		// Update the parameters
		xtxm := mat.NewDense(nvar, nvar, xtx)
		xtyv := mat.NewVecDense(nvar, xty)
		err := nparam.SolveVec(xtxm, xtyv)
		if err != nil {
			for j := 0; j < nvar; j++ {
				fmt.Printf("%8d %12.4f %12.4f\n", j, xty[j], xtx[j*nvar+j])
			}
			panic(err)
		}
		params = nparam.RawVector().Data

		// Check convergence
		dev = append(dev, devi)
		if len(dev) > 3 && math.Abs(dev[len(dev)-1]-dev[len(dev)-2]) < dtol {
			break
		}

		if glm.log != nil {
			msg := fmt.Sprintf("Iteration %d: deviance=%.10f\n", iter+1, devi)
			glm.log.Print(msg)
		}
	}

	if glm.log != nil {
		glm.log.Print("IRLS converged\n")
	}

	glm.putNslice(linpred)
	glm.putNslice(mn)
	glm.putNslice(va)
	glm.putNslice(lderiv)
	glm.putNslice(irlsw)
	glm.putNslice(adjy)

	return params
}

func (glm *GLM) irlsXprod(xdat [][]statmodel.Dtype, adjy, irlsw, xty, xtx []float64) {

	if len(adjy) >= glm.concurrentIRLS {
		glm.irlsXprodConcurrent(xdat, adjy, irlsw, xty, xtx)
		return
	}

	nvar := len(xdat)

	for j1 := range glm.xpos {

		// Update x' w^-1 yadj
		xda := xdat[j1]
		var u float64
		for i := range adjy {
			u += adjy[i] * float64(xda[i]) * irlsw[i]
		}
		xty[j1] += u

		// Update x' w^-1 x
		for j2 := 0; j2 <= j1; j2++ {
			xdb := xdat[j2]
			var u float64
			for i := range xda {
				u += float64(xda[i]*xdb[i]) * irlsw[i]
			}
			xtx[j1*nvar+j2] += u
		}
	}
}

// irlsXprodConcurrent is a concurrent version of irlsXprod
func (glm *GLM) irlsXprodConcurrent(xdat [][]statmodel.Dtype, adjy, irlsw, xty, xtx []float64) {

	nvar := len(xdat)

	var wg sync.WaitGroup

	for j1 := range glm.xpos {

		// Update x' w^-1 yadj
		xda := xdat[j1]
		wg.Add(1)
		go func(j1 int) {
			var u float64
			for i := range adjy {
				u += adjy[i] * float64(xda[i]) * irlsw[i]
			}
			xty[j1] += u
			wg.Done()
		}(j1)

		// Update x' w^-1 x
		for j2 := 0; j2 <= j1; j2++ {
			xdb := xdat[j2]
			wg.Add(1)
			go func(j1, j2 int) {
				var u float64
				for i := range xda {
					u += float64(xda[i]*xdb[i]) * irlsw[i]
				}
				xtx[j1*nvar+j2] += u
				wg.Done()
			}(j1, j2)
		}
	}

	wg.Wait()
}

func (glm *GLM) startingMu(y []statmodel.Dtype, mn []float64) {

	var q float64
	name := strings.ToLower(glm.fam.Name)
	if name == "binomial" {
		q = 0.5
	} else {
		for i := range y {
			q += float64(y[i])
		}
		q /= float64(len(y))
	}
	for i := range mn {
		mn[i] = (float64(y[i]) + q) / 2
		if mn[i] < 0.1 {
			mn[i] = 0.1
		}
	}
}
