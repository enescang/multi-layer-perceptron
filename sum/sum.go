package sum

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/enescang/multi-layer-perceptron/mlp"
)

func Run() {
	var sumMlp *mlp.MLP
	sumMlp = &mlp.MLP{}
	sumMlp.Layers = &[]mlp.Layer{}
	sumMlp.LearningRate = 0.01
	sumMlp.ErrorRate = 5

	i := RandomLayer(2, 20)
	i.IsInput = true
	l2 := RandomLayer(20, 45)
	l3 := RandomLayer(45, 15)
	l4 := RandomLayer(15, 1)
	o := RandomLayer(1, 0)
	o.IsOutput = true

	*sumMlp.Layers = append(*sumMlp.Layers, i, l2, l3, l4, o)
	byteArr, err := os.ReadFile("./sum_mlp_struct.json")
	if err != nil {
		fmt.Println("Read File Error:", err)
		return
	}
	sumMlp.FromJSON(string(byteArr))
	sumMlp.Build()
	sumMlp.Inputs = &[]mlp.InputRow{}

	/*
		i1 := SumRow(0.1, 0.1, 0.2)
		i2 := SumRow(0.1, 0.4, 0.5)
		i3 := SumRow(0.7, 0.2, 0.9)
		i4 := SumRow(0.12, 0.12, 0.24)
		i5 := SumRow(0.1, 0.8, 0.9)
		fmt.Println(i1, i2, i3, i4, i5)
	*/

	for i := 0; i < 100; i++ {
		r := SumRow(float64(rand.Float64()), float64(rand.Float64()))
		fmt.Println(r)
		*sumMlp.Inputs = append(*sumMlp.Inputs, r)
	}

	start_at := time.Now()
	for {
		completed := sumMlp.Iteration()
		if completed {
			break
		}
	}
	fmt.Println("Sum duration:", time.Since(start_at))

	testRow := SumRow(1, 0.9)
	result := sumMlp.IterateWithRow(testRow)
	fmt.Println(result)

	jsonStr, err := sumMlp.ToJSON()
	if err != nil {
		fmt.Println(err)
	}
	os.WriteFile("./sum_mlp_struct.json", []byte(jsonStr), 0644)
}

func RandomCell(cell_count int, weight_size int) mlp.Cell {
	cell2 := mlp.Cell{
		Name:           "Random Cell",
		Value:          0,
		ValueDelta:     0,
		OutsideWeights: &[]mlp.CellWeight{},
	}
	for i := 0; i < weight_size; i++ {
		var t float64
		t = rand.Float64()
		if rand.Float64() >= 0.5 {
			t = t * -1
		}
		*cell2.OutsideWeights = append(*cell2.OutsideWeights, mlp.CellWeight{Value: t})
	}
	return cell2
}

func RandomLayer(cell_count int, weight_count int) mlp.Layer {
	var layer mlp.Layer
	for i := 0; i < cell_count; i++ {
		layer.AddCell(RandomCell(cell_count, weight_count))
	}
	return layer
}

func SumRow(a ...float64) mlp.InputRow {
	var total float64
	for _, v := range a {
		total += v
	}
	total = total * 0.1
	return mlp.InputRow{Inputs: a, Expecteds: []float64{total}}
}
