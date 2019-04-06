package duration

import "testing"

func TestConcordance1(t *testing.T) {

	time := []float64{1, 2, 3, 4, 5, 6}
	status := []float64{1, 1, 1, 1, 1, 1}
	score := []float64{7, 6, 5, 4, 3, 2}

	c := NewConcordance(time, status, score).Done()
	if c.Concordance(100) != 1 {
		t.Fail()
	}
}
