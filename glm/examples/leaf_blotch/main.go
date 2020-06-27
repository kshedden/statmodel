/*
This example follows the leaf blotch analysis in McCullagh and
Nelder's GLM book.

The data are proportions between 0 and 1, arranged in a complete two-way
layout.  The mean model is an additive factorial model.  The
parameters are fit using a binomial GLM with the usual logit link
function.  This is a quasi-likelihood analysis since the data are not
binary.

The first model fit below uses the default binomial variance function.
This produces a very small scale parameter estimate, and the
standardized residuals do not have constant variance relative to the
fitted mean.

The second model fit below uses a variance function that is the square
of the usual binomial variance function.  This variance function gives
standardized residuals that are roughly constant with respect to the
mean, and the scale parameter estimate is close to 1.

Residual / mean plots are constructed to show how the specification
of the GLM variance function impacts the residual distribution.

The analysis follows the SAS manual:

https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_glimmix_sect016.htm
*/

package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"strconv"

	"github.com/kshedden/statmodel/glm"
	"github.com/kshedden/statmodel/statmodel"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

var (
	raw string = `0.05,0.00,1.25,2.50,5.50,1.00,5.00,5.00,17.50
0.00,0.05,1.25,0.50,1.00,5.00,0.10,10.00,25.00
0.00,0.05,2.50,0.01,6.00,5.00,5.00,5.00,42.50
0.10,0.30,16.60,3.00,1.10,5.00,5.00,5.00,50.00
0.25,0.75,2.50,2.50,2.50,5.00,50.00,25.00,37.50
0.05,0.30,2.50,0.01,8.00,5.00,10.00,75.00,95.00
0.50,3.00,0.00,25.00,16.50,10.00,50.00,50.00,62.50
1.30,7.50,20.00,55.00,29.50,5.00,25.00,75.00,95.00
1.50,1.00,37.50,5.00,20.00,50.00,50.00,75.00,95.00
1.50,12.70,26.25,40.00,43.50,75.00,75.00,75.00,95.00`

	// This is the square of the usual binomial variance function.
	squaredbinom = &glm.Variance{
		Name: "SquaredBinomial",
		Var: func(mn, va []float64) {
			for i := range mn {
				va[i] = mn[i] * mn[i] * (1 - mn[i]) * (1 - mn[i])
			}
		},
		Deriv: func(mn, va []float64) {
			for i := range mn {
				va[i] = 2*mn[i] - 6*mn[i]*mn[i] + 4*mn[i]*mn[i]*mn[i]
			}
		},
	}
)

// setup builds a dataset from the raw data.
func setup() (statmodel.Dataset, []string) {

	rdr := bytes.NewReader([]byte(raw))
	rdc := csv.NewReader(rdr)

	// There is one outcome variable, 10 row effects, and 9 column effects
	nrow := 10
	ncol := 9

	// The outcome variable
	var y []float64

	// Row and column indicators
	rowix := make([][]float64, nrow)
	colix := make([][]float64, ncol)

	for row := 0; ; row++ {
		rec, err := rdc.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}

		for col := range rec {
			x, err := strconv.ParseFloat(rec[col], 64)
			if err != nil {
				panic(err)
			}

			// Convert percent to proportion
			y = append(y, x/100)

			// Row indicators
			for j := 0; j < nrow; j++ {
				if j == row {
					rowix[j] = append(rowix[j], 1)
				} else {
					rowix[j] = append(rowix[j], 0)
				}
			}

			// Column indicators
			for j := 0; j < ncol; j++ {
				if j == col {
					colix[j] = append(colix[j], 1)
				} else {
					colix[j] = append(colix[j], 0)
				}
			}
		}
	}

	da := [][]float64{y}
	da = append(da, rowix...)
	da = append(da, colix[0:ncol-1]...) // Omit the final column indicator

	varnames := []string{"y"}

	for j := 0; j < nrow; j++ {
		vn := fmt.Sprintf("row%d", j)
		varnames = append(varnames, vn)
	}

	for j := 0; j < ncol-1; j++ {
		vn := fmt.Sprintf("col%d", j)
		varnames = append(varnames, vn)
	}

	xnames := varnames[1:]

	return statmodel.NewDataset(da, varnames), xnames
}

func residPlot(lp, resid []float64, title, filename string) {

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = title
	p.X.Label.Text = "Linear predictor"
	p.Y.Label.Text = "Pearson residual"

	pts := make(plotter.XYs, len(lp))
	for i := range lp {
		pts[i].X = lp[i]
		pts[i].Y = resid[i]
	}

	err = plotutil.AddScatters(p, pts)
	if err != nil {
		panic(err)
	}

	err = p.Save(6*vg.Inch, 4*vg.Inch, filename)
	if err != nil {
		panic(err)
	}
}

func main() {

	data, xnames := setup()

	// Initial model, the scale parameter estimate is around 0.09.
	c := glm.DefaultConfig()
	c.Family = glm.NewFamily(glm.BinomialFamily)
	c.DispersionForm = glm.DispersionFree
	model, err := glm.NewGLM(data, "y", xnames, c)
	if err != nil {
		panic(err)
	}
	result := model.Fit()

	fmt.Printf("%v\n", result.Summary())

	residPlot(result.LinearPredictor(nil), result.PearsonResid(nil),
		"Default variance", "defvar.pdf")
	fmt.Printf("%v\n", result.Summary())

	// Model with squared variance function, the scale parameter estimate is close to 1.
	c = glm.DefaultConfig()
	c.Family = glm.NewFamily(glm.BinomialFamily)
	c.DispersionForm = glm.DispersionFree
	c.VarFunc = squaredbinom
	model, err = glm.NewGLM(data, "y", xnames, c)
	if err != nil {
		panic(err)
	}
	result = model.Fit()

	residPlot(result.LinearPredictor(nil), result.PearsonResid(nil),
		"Squared variance", "sqvar.pdf")
	fmt.Printf("%v\n", result.Summary())
}
