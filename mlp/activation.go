package mlp

import "math"

func Sigmoid(n float64) float64 {
	return 1 / (1 + math.Exp(-n))
}
