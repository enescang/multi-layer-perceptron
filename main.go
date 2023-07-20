package main

import (
	"github.com/enescang/multi-layer-perceptron/mlp"
)

func CreateMLP() *mlp.MLP {
	testMlp := mlp.MLP{}
	return &testMlp
}

func main() {
	testMlp := CreateMLP()
	testMlp.LearningRate = 1
	testMlp.ErrorRate = 0.01
	inputLayer := HandleInputLayer()
	layer2 := HandleLayer2()
	outputLayer := HandleOutputLayer()

	testMlp.Layers = &[]mlp.Layer{*inputLayer, *layer2, *outputLayer}
	testMlp.Build()
	testMlp.PrintBuildedLayers()

	testMlp.Iteration()
}

func HandleInputLayer() *mlp.Layer {
	layer1 := mlp.Layer{}
	layer1.IsInput = true
	layer1.Name = "INPUT LAYER"
	cell1 := mlp.Cell{
		Name:       "Input Layer Cell 1",
		Value:      0.35,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.1},
			{Value: 0.4},
		},
	}
	cell2 := mlp.Cell{
		Name:       "Input Layer Cell 2",
		Value:      0.9,
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
		},
	}
	cell2 := mlp.Cell{
		Name:       "Second Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.9},
		},
	}
	layer1.AddCell(cell1)
	layer1.AddCell((cell2))
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
		Expected:       0.5,
		OutsideWeights: &[]mlp.CellWeight{},
	}
	layer1.AddCell(cell1)
	return &layer1
}
