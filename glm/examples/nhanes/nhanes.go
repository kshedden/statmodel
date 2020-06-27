// +build ignore

// This script contains some examples of fitting GLM's to NHANES data.
//
// The dstream Go package is used to prepare the data for analysis.
//
// Prior to running this script, download the NHANES demographics (DEMO_G.XPT)
// and blood pressure (BPX_G.XPT) data from here:
//
// https://wwwn.cdc.gov/Nchs/Nhanes/Search/DataPage.aspx?Component=Examination&CycleBeginYear=2011
//
// This script uses a merged dataset in csv format.  Since Go doesn't seem to
// currently have a reader for SAS XPT files, to obtain a csv version of
// the NHANES dataset run the following Python program:
//
//  # Python script below, requires Pandas
//  import pandas as pd
//
//  fn1 = "DEMO_G.XPT"
//  fn2 = "BPX_G.XPT"
//
//  ds1 = pd.read_sas(fn1)
//  ds2 = pd.read_sas(fn2)
//
//  ds = pd.merge(ds1, ds2, left_on="SEQN", right_on="SEQN")
//
//  ds.to_csv("nhanes.csv.gz", index=False, compression="gzip")

package main

import (
	"compress/gzip"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/kshedden/formula"
	"github.com/kshedden/statmodel/glm"
)

var (
	source formula.DataSource
)

// toString converts a float64 slice (containing whole numbers)
// to a string slice.  For a variable to be treated as a categorical
// variable by the GLM, it must be represented with strings.
func toString(x []float64) []string {
	y := make([]string, len(x))
	for i, v := range x {
		y[i] = fmt.Sprintf("%.0f", v)
	}
	return y
}

// init loads the NHANES data from a csv file.
func init() {

	fid, err := os.Open("nhanes.csv.gz")
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	gid, err := gzip.NewReader(fid)
	if err != nil {
		panic(err)
	}
	defer gid.Close()

	rdr := csv.NewReader(gid)

	names, err := rdr.Read()
	if err != nil {
		panic(err)
	}

	data := make([][]float64, len(names))
	for {
		row, err := rdr.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}

		for j, x := range row {
			y, err := strconv.ParseFloat(x, 64)
			if err != nil {
				y = math.NaN()
			}
			data[j] = append(data[j], y)
		}
	}

	// These variables are converted to strings, so that
	// they are treated as categorical variables in the models
	// to be fit below.
	tostring := map[string]bool{"RIDRETH1": true}

	var datax []interface{}
	for j, x := range data {
		if tostring[names[j]] {
			datax = append(datax, toString(x))
		} else {
			datax = append(datax, x)
		}
	}

	source = formula.NewSource(names, datax)
}

func model1() {

	msg := `
Linear regression (ordinary least squares) for systolic blood pressure,
using two predictor variables: gender (RIAGENDR) and age (RIDAGEYR).
Gender is treated as a quantitative variable and is coded as 1 for
males and 2 for females.
`
	fml := []string{"BPXSY1", "1 + RIAGENDR + RIDAGEYR"}

	f, err := formula.NewMulti(fml, source, nil)
	if err != nil {
		panic(err)
	}

	da, err := f.Parse()
	if err != nil {
		panic(err)
	}
	da = da.DropNA()

	xnames := []string{"icept", "RIAGENDR", "RIDAGEYR"}

	model, err := glm.NewGLM(da, "BPXSY1", xnames, nil)
	if err != nil {
		panic(err)
	}
	rslt := model.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model2() {

	msg := `
Linear regression (ordinary least squares) for systolic blood pressure,
including ethnicity as a categorical covariate, using level 5 (other
race/multiracial) as the reference category.
`

	fml := []string{"BPXSY1", "1 + RIAGENDR + RIDAGEYR + RIDRETH1"}

	f, err := formula.NewMulti(fml, source, nil)
	if err != nil {
		panic(err)
	}

	da, err := f.Parse()
	if err != nil {
		panic(err)
	}
	da = da.DropNA()

	xnames := []string{"icept", "RIAGENDR", "RIDAGEYR", "RIDRETH1[1]", "RIDRETH1[2]", "RIDRETH1[3]", "RIDRETH1[4]"}

	model, err := glm.NewGLM(da, "BPXSY1", xnames, nil)
	if err != nil {
		panic(err)
	}
	rslt := model.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model3() {

	msg := `
Linear regression (ordinary least squares) for systolic blood pressure,
including gender, age, ethnicity, and the interaction between gender
and age as covariates.  Ethnicity is a categorical covariate with level
5 (other race/multiracial) as the reference category.
`

	fml := []string{"BPXSY1", "1 + RIAGENDR + RIDAGEYR + RIDRETH1 + RIAGENDR * RIDAGEYR"}

	f, err := formula.NewMulti(fml, source, nil)
	if err != nil {
		panic(err)
	}

	da, err := f.Parse()
	if err != nil {
		panic(err)
	}
	da = da.DropNA()

	xnames := []string{"icept", "RIAGENDR", "RIDAGEYR", "RIDRETH1[1]", "RIDRETH1[2]", "RIDRETH1[3]", "RIDRETH1[4]",
		"RIAGENDR:RIDAGEYR"}

	model, err := glm.NewGLM(da, "BPXSY1", xnames, nil)
	if err != nil {
		panic(err)
	}
	rslt := model.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model4() {

	msg := `
Regularized least squares regression (Lasso regression) for systolic
blood pressure, using equal penalty weights for all covariates and
zero penalty for the intercept.
`

	fml := []string{"BPXSY1", "1 + RIAGENDR + RIDAGEYR + RIDRETH1"}

	f, err := formula.NewMulti(fml, source, nil)
	if err != nil {
		panic(err)
	}

	da, err := f.Parse()
	if err != nil {
		panic(err)
	}
	da = da.DropNA()

	l1pen := make(map[string]float64)
	for _, v := range da.Names() {
		if v != "icept" {
			l1pen[v] = 0.01
		}
	}
	conf := glm.DefaultConfig()
	conf.L1Penalty = l1pen

	xnames := []string{"icept", "RIAGENDR", "RIDAGEYR", "RIDRETH1[1]", "RIDRETH1[2]", "RIDRETH1[3]", "RIDRETH1[4]"}

	model, err := glm.NewGLM(da, "BPXSY1", xnames, conf)
	if err != nil {
		panic(err)
	}
	rslt := model.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

func model5() {

	msg := `
Linear regression with systolic blood pressure as the outcome,
using a square root transform in the formula.
`

	fml := []string{"BPXSY1", "1 + RIAGENDR + sqrt(RIDAGEYR) + RIDRETH1"}

	funcs := make(map[string]formula.Func)
	funcs["sqrt"] = func(na string, x []float64) *formula.ColSet {
		y := make([]float64, len(x))
		for i, v := range x {
			y[i] = v * v
		}
		return formula.NewColSet([]string{na}, [][]float64{y})
	}

	conf := &formula.Config{Funcs: funcs}
	f, err := formula.NewMulti(fml, source, conf)
	if err != nil {
		panic(err)
	}

	da, err := f.Parse()
	if err != nil {
		panic(err)
	}
	da = da.DropNA()

	xnames := []string{"icept", "RIAGENDR", "sqrt(RIDAGEYR)", "RIDRETH1[1]", "RIDRETH1[2]", "RIDRETH1[3]", "RIDRETH1[4]"}

	model, err := glm.NewGLM(da, "BPXSY1", xnames, nil)
	if err != nil {
		panic(err)
	}
	rslt := model.Fit()

	fmt.Printf(msg + "\n")
	fmt.Printf(rslt.Summary().String() + "\n\n")
}

// binfunc dichotomizes blood pressure to 1 (>= 130) and
// 0 (< 130).
func binfunc(na string, x []float64) *formula.ColSet {
	y := make([]float64, len(x))
	for i, v := range x {
		if v >= 130 {
			y[i] = 1
		}
	}
	return formula.NewColSet([]string{na}, [][]float64{y})
}

func model6() {

	msg := `
Logistic regression using high blood pressure status (binary) as
the dependent variable, and gender and age as predictors.
`

	funcs := make(map[string]formula.Func)
	funcs["bin"] = binfunc

	fml := []string{"bin(BPXSY1)", "1 + RIAGENDR + RIDAGEYR"}

	conf := &formula.Config{Funcs: funcs}
	f, err := formula.NewMulti(fml, source, conf)
	if err != nil {
		panic(err)
	}

	da, err := f.Parse()
	if err != nil {
		panic(err)
	}
	da = da.DropNA()

	xnames := []string{"icept", "RIAGENDR", "RIDAGEYR"}

	c := glm.DefaultConfig()
	c.Family = glm.NewFamily(glm.BinomialFamily)
	model, err := glm.NewGLM(da, "bin(BPXSY1)", xnames, c)
	if err != nil {
		panic(err)
	}
	rslt := model.Fit()

	smry := rslt.Summary()
	fmt.Printf(msg + "\n")
	fmt.Printf(smry.String() + "\n\n")

	smry = smry.SetScale(math.Exp, "Parameters are shown as odds ratios")

	fmt.Printf(smry.String() + "\n\n")
}

func model7() {

	msg := `
Elastic net penalized logistic regression for high blood pressure
status, with L1 and L2 penalties.  Age and gender are the predictor
variables.
`

	funcs := make(map[string]formula.Func)
	funcs["bin"] = binfunc

	fml := []string{"bin(BPXSY1)", "1 + RIAGENDR + RIDAGEYR"}

	conf := &formula.Config{Funcs: funcs}
	f, err := formula.NewMulti(fml, source, conf)
	if err != nil {
		panic(err)
	}

	da, err := f.Parse()
	if err != nil {
		panic(err)
	}
	da = da.DropNA()

	xnames := []string{"icept", "RIAGENDR", "RIDAGEYR"}

	c := glm.DefaultConfig()
	c.Family = glm.NewFamily(glm.BinomialFamily)
	c.L1Penalty = map[string]float64{"RIAGENDR": 1}
	c.L2Penalty = map[string]float64{"RIAGENDR": 0.01, "RIDAGEYR": 0.01}

	model, err := glm.NewGLM(da, "bin(BPXSY1)", xnames, c)
	if err != nil {
		panic(err)
	}
	rslt := model.Fit()
	smry := rslt.Summary()

	fmt.Printf(msg + "\n")
	fmt.Printf(smry.String() + "\n\n")
}

func main() {
	model1()
	model2()
	model3()
	model4()
	model5()
	model6()
	model7()
}
