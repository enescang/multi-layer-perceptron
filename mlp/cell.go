package mlp

type Cell struct {
	Name           string
	Value          float64
	ValueDelta     float64
	ErrorDelta     float64
	Expected       float64
	OutsideWeights *[]CellWeight
	InsideWeights  *[]CellWeight
}

type CellWeight struct {
	Value      float64
	ValueDelta float64
}
