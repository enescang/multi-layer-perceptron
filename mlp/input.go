package mlp

type InputRow struct {
	Inputs    []float64
	Expecteds []float64
}

type IterationResult struct {
	Passed  bool
	Outputs []OutputResult
}

type OutputResult struct {
	Output   float64
	Expected float64
	Error    float64
}
