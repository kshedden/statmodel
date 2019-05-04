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
	"fmt"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/dstream/formula"
	"github.com/kshedden/statmodel/glm"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

var (
	data dstream.Dstream

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
	squaredbinom glm.Variance = glm.Variance{
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

func setup() dstream.Dstream {

	rdr := bytes.NewReader([]byte(raw))

	types := []dstream.VarType{
		{"pct1", dstream.Float64},
		{"pct2", dstream.Float64},
		{"pct3", dstream.Float64},
		{"pct4", dstream.Float64},
		{"pct5", dstream.Float64},
		{"pct6", dstream.Float64},
		{"pct7", dstream.Float64},
		{"pct8", dstream.Float64},
		{"pct9", dstream.Float64},
	}

	da := dstream.FromCSV(rdr).SetTypes(types).Done()

	var y []float64
	var row, col []string
	var irow int
	for da.Next() {

		var z []float64
		for j := 0; j < 9; j++ {
			z = da.GetPos(j).([]float64)
			for i := range z {
				row = append(row, fmt.Sprintf("%d", irow+i))
				col = append(col, fmt.Sprintf("%d", j))
				y = append(y, z[i]/100)
			}
		}
		irow += len(z)
	}

	dx := []interface{}{row, col, y}
	dz := dstream.NewFromFlat(dx, []string{"row", "col", "y"})
	dz = formula.New("row + col", dz).Keep("y").Done()
	dz = dstream.MemCopy(dz, true)

	return dstream.DropCols(dz, "col[8]")
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

	data := setup()

	// Initial model, the scale parameter estimate is around 0.09.
	model := glm.NewGLM(data, "y").Family(glm.NewFamily(glm.BinomialFamily))
	model = model.DispersionForm(glm.DispersionFree)
	model = model.Done()
	result := model.Fit()
	residPlot(result.LinearPredictor(nil), result.PearsonResid(nil),
		"Default variance", "defvar.pdf")
	fmt.Printf("%v\n", result.Summary())

	// Initial model, the scale parameter estimate is close to 1.
	model = glm.NewGLM(data, "y").Family(glm.NewFamily(glm.BinomialFamily))
	model = model.DispersionForm(glm.DispersionFree)
	model = model.VarFunc(&squaredbinom)
	model = model.Done()
	result = model.Fit()
	residPlot(result.LinearPredictor(nil), result.PearsonResid(nil),
		"Squared variance", "sqvar.pdf")
	fmt.Printf("%v\n", result.Summary())
}
