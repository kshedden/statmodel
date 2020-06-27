// This script demonstrates fitting a proportional hazards
// regression model, using the 'diabetic' data from the R
// survival package

package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/kshedden/statmodel/duration"
	"github.com/kshedden/statmodel/statmodel"
)

func readData(side string) statmodel.Dataset {

	fid, err := os.Open("diabetic.csv")
	if err != nil {
		panic(err)
	}
	defer fid.Close()

	rd := csv.NewReader(fid)

	names, err := rd.Read()
	if err != nil {
		panic(err)
	}

	da := make([][]float64, 8)

	for {
		row, err := rd.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}

		if row[3] != side {
			continue
		}

		// Handle numeric columns
		for _, j := range []int{0, 2, 4, 5, 6, 7} {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				panic(err)
			}
			da[j] = append(da[j], v)
		}

		if row[1] == "argon" {
			da[1] = append(da[1], 1)
		} else {
			da[1] = append(da[1], 0)
		}
	}

	return statmodel.NewDataset(da, names)
}

func main() {

	data := readData("left")
	xnames := []string{"laser", "age", "trt", "risk"}

	model, err := duration.NewPHReg(data, "time", "status", xnames, nil)
	if err != nil {
		panic(err)
	}

	result, err := model.Fit()
	if err != nil {
		panic(err)
	}

	fmt.Printf("%v\n", result.Summary())
}
