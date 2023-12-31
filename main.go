package main

import (
	"fmt"
	"time"

	"github.com/enescang/multi-layer-perceptron/mlp"
)

func CreateMLP() *mlp.MLP {
	testMlp := mlp.MLP{}
	return &testMlp
}

func main() {
	testMlp := CreateMLP()
	testMlp.LearningRate = 1
	testMlp.ErrorRate = 0.03

	inputLayer := HandleInputLayer()
	layer2 := HandleLayer2()
	outputLayer := HandleOutputLayer()
	testMlp.Inputs = GetInputRows()

	testMlp.Layers = &[]mlp.Layer{*inputLayer, *layer2, *HandleLayer3(), *outputLayer}
	testMlp.Build()
	//testMlp.PrintBuildedLayers()

	start_at := time.Now()
	for {
		completed := testMlp.Iteration()
		if completed {
			break
		}
	}
	duration := time.Since(start_at)
	fmt.Println("TRAIN DURATION", duration)
	result := testMlp.IterateWithRow(mlp.InputRow{Inputs: []float64{0.8, 0.1}, Expecteds: []float64{0.9}})
	fmt.Println(result)
}

func GetInputRows() *[]mlp.InputRow {
	rows := []mlp.InputRow{}
	row1 := mlp.InputRow{Inputs: []float64{0.1, 0.1}, Expecteds: []float64{0.2}}
	row2 := mlp.InputRow{Inputs: []float64{0.11, 0.2}, Expecteds: []float64{0.31}}
	row3 := mlp.InputRow{Inputs: []float64{0.23, 0.05}, Expecteds: []float64{0.28}}
	row4 := mlp.InputRow{Inputs: []float64{0.32, 0.1}, Expecteds: []float64{0.42}}
	row5 := mlp.InputRow{Inputs: []float64{0.17, 0.115}, Expecteds: []float64{0.285}}
	row6 := mlp.InputRow{Inputs: []float64{0.2, 0.2}, Expecteds: []float64{0.4}}
	row7 := mlp.InputRow{Inputs: []float64{0.23, 0.5}, Expecteds: []float64{0.73}}
	row8 := mlp.InputRow{Inputs: []float64{0.63, 0.37}, Expecteds: []float64{1}}
	rows = append(rows, row1, row2, row3, row4, row5, row6, row7, row8)
	return &rows
}

func HandleInputLayer() *mlp.Layer {
	layer1 := mlp.Layer{}
	layer1.IsInput = true
	layer1.Name = "INPUT LAYER"
	cell1 := mlp.Cell{
		Name:       "Input Layer Cell 1",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.1},
			{Value: 0.4},
		},
	}
	cell2 := mlp.Cell{
		Name:       "Input Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.8},
			{Value: 0.6},
		},
	}
	layer1.AddCell(cell1)
	layer1.AddCell((cell2))
	return &layer1
}

func HandleLayer2() *mlp.Layer {
	layer1 := mlp.Layer{}
	layer1.Name = "LAYER 2"
	cell1 := mlp.Cell{
		Name:       "Second Layer Cell 1",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.3},
			{Value: 0.44},
			{Value: 0.8},
		},
	}
	cell2 := mlp.Cell{
		Name:       "Second Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.9},
			{Value: 0.76},
			{Value: 0.09001},
		},
	}
	layer1.AddCell(cell1)
	layer1.AddCell((cell2))
	return &layer1
}

func HandleLayer3() *mlp.Layer {
	layer1 := mlp.Layer{}
	layer1.Name = "LAYER 3"
	cell1 := mlp.Cell{
		Name:       "Third Layer Cell 1",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.96},
		},
	}
	cell2 := mlp.Cell{
		Name:       "Third Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.032},
		},
	}
	cell3 := mlp.Cell{
		Name:       "Third Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.092},
		},
	}
	layer1.AddCell(cell1)
	layer1.AddCell(cell2)
	layer1.AddCell(cell3)
	return &layer1
}

func HandleOutputLayer() *mlp.Layer {
	layer1 := mlp.Layer{}
	layer1.Name = "OUTPUT LAYER"
	layer1.IsOutput = true
	cell1 := mlp.Cell{
		Name:           "Output Layer Cell 1",
		Value:          0,
		ValueDelta:     0,
		ErrorDelta:     0,
		Expected:       0,
		OutsideWeights: &[]mlp.CellWeight{},
	}
	layer1.AddCell(cell1)
	return &layer1
}
