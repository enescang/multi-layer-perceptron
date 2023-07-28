package mlp

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

type MLP struct {
	Layers         *[]Layer
	Inputs         *[]InputRow
	LearningRate   float64
	ErrorRate      float64
	IterationCount int
}

func (mlp *MLP) Build() {
	var acc *Layer
	for index := range *mlp.Layers {
		var l *Layer
		l = &(*mlp.Layers)[index]
		if acc != nil {
			l.Prev = acc
			acc.Next = l
		}
		acc = &(*mlp.Layers)[index]
	}
}

func (mlp *MLP) PrintBuildedLayers() {
	for _, v := range *mlp.Layers {
		fmt.Println("CURRENT:", v.Name)
		if v.Prev != nil {
			fmt.Println("PREV:", v.Prev.Name)
		}
		if v.Next != nil {
			fmt.Println("NEXT:", v.Next.Name)
		}

		fmt.Println("END ****")
	}
}

func (mlp *MLP) Iteration() (completed bool) {
	var correct int
	mlp.IterationCount++
	mlp.ShuffleInputs()
	for i, row := range *mlp.Inputs {
		//start_at := time.Now()
		iteration_result := mlp.IterateWithRow(row)
		if i%5 == 0 {
			//fmt.Println("ONE ROW ITERATION DONE", time.Since(start_at), i, iteration_result)
		}
		if iteration_result.Passed {
			correct++
		}
	}
	if mlp.IterationCount%100 == 0 {
		fmt.Println("ITERATION:", mlp.IterationCount, " TOTAL CORRECT:", correct, float64(correct)/float64(len(*mlp.Inputs))*100)
	}
	if mlp.IterationCount > 3000 && correct >= len(*mlp.Inputs)*90/100 || (mlp.IterationCount == 10 && false) {
		fmt.Println("ITERATION DONE!", mlp.IterationCount)
		completed = true
		return completed
	}
	completed = false
	return completed
}

func (mlp *MLP) IterateWithRow(row InputRow) (iteration_result IterationResult) {
	mlp.prepareInputAndOutputLayers(row)
	iteration_result = mlp.calculateHiddenLayerCellValues()
	if iteration_result.Passed == false {
		//fmt.Println("USE BACKPROPAGATION")
		mlp.CalculateErrorDelta()
		mlp.CalculateNewWeights()
		return iteration_result
	}
	return iteration_result
}

func (mlp *MLP) prepareInputAndOutputLayers(row InputRow) {
	var inputLayerPointer *Layer
	var outputLayerPointer *Layer
	for layer_index, layer_val := range *mlp.Layers {
		if layer_val.IsInput {
			inputLayerPointer = &(*mlp.Layers)[layer_index]
		}
		if layer_val.IsOutput {
			outputLayerPointer = &(*mlp.Layers)[layer_index]
		}
		if inputLayerPointer != nil && outputLayerPointer != nil {
			break
		}
	}

	if inputLayerPointer == nil {
		log.Fatal("Error: mlp.PrepareInputLayer. Input Layer Not Found")
		return
	}
	if outputLayerPointer == nil {
		log.Fatal("Error: mlp.PrepareInputLayer. Output Layer Not Found")
		return
	}

	for cell_index := range *inputLayerPointer.Cells {
		(*inputLayerPointer.Cells)[cell_index].Value = row.Inputs[cell_index]
	}

	for cell_index := range *outputLayerPointer.Cells {
		(*outputLayerPointer.Cells)[cell_index].Expected = row.Expecteds[cell_index]
	}
}

func (mlp *MLP) calculateHiddenLayerCellValues() (result IterationResult) {
	var pointerLayer *Layer
	var pointerPrevLayer *Layer
	var pointerCell *Cell
	for layer_index := range *mlp.Layers {
		pointerLayer = &(*mlp.Layers)[layer_index]
		if pointerLayer.IsInput {
			continue
		}
		if pointerLayer.Prev == nil {
			log.Fatal("There is a problem in STRUCT of MLP. Current Layer does NOT have Previous Layer.")
			break
		}

		pointerPrevLayer = pointerLayer.Prev
		for cell_index := range *pointerLayer.Cells {
			pointerCell = &(*pointerLayer.Cells)[cell_index]
			var new_value float64 = 0
			for prev_layer_cell_index := range *pointerPrevLayer.Cells {
				var prev_layer_current_cell *Cell
				prev_layer_current_cell = &(*pointerPrevLayer.Cells)[prev_layer_cell_index]
				cell_outside_weight_value := (*prev_layer_current_cell.OutsideWeights)[cell_index].Value
				//fmt.Println("WEIGHT:", cell_outside_weight_value, " CUURENT VALUE", prev_layer_current_cell.Value, pointerPrevLayer.Name)
				new_value += (cell_outside_weight_value * prev_layer_current_cell.Value)
			}
			//fmt.Println("BEFORE Activation:", new_value)
			new_value = Sigmoid(new_value)
			//fmt.Println("AFTER Activation:", new_value)

			pointerCell.ValueDelta = new_value
			pointerCell.Value = new_value
			if pointerLayer.IsOutput {
				var err_value float64 = pointerCell.Expected - pointerCell.Value
				output_result := OutputResult{Output: pointerCell.Value, Expected: pointerCell.Expected, Error: err_value}
				result.Outputs = append(result.Outputs, output_result)
				//fmt.Println("ERROR:", err_value, "%", (pointerCell.Expected * mlp.ErrorRate / 100))
				if math.Abs(err_value) > (pointerCell.Expected * mlp.ErrorRate / 100) {
					result.Passed = false
				} else {
					result.Passed = true
				}
			}
			//fmt.Println(pointerCell.Name, "delta VALUE:", pointerCell.ValueDelta)
		}
	}
	return result
}

func (mlp *MLP) CalculateErrorDelta() {
	var pointerLayer *Layer
	for i := len(*mlp.Layers) - 1; i >= 0; i-- {
		pointerLayer = &(*mlp.Layers)[i]
		if pointerLayer.IsInput {
			continue
		}
		if pointerLayer.IsOutput {
			for cell_index := range *pointerLayer.Cells {
				var cell *Cell
				cell = &(*pointerLayer.Cells)[cell_index]
				var error_delta float64 = 0
				error_delta = cell.Value * (1 - cell.Value) * (cell.Expected - cell.Value)
				cell.ErrorDelta = error_delta
				//fmt.Println("OUTPUT ERROR:", error_delta, "cell value:", cell.Value, "expected:", cell.Expected)
			}
			continue
		}
		for cell_index := range *pointerLayer.Cells {
			var cell *Cell
			cell = &(*pointerLayer.Cells)[cell_index]
			var error_delta float64 = 0
			error_delta = cell.Value * (1 - cell.Value)

			var weight_multiplation float64 = 0
			for weight_index := range *cell.OutsideWeights {
				var outside_weight *CellWeight
				outside_weight = &(*cell.OutsideWeights)[weight_index]
				weight_multiplation += (outside_weight.Value * (*pointerLayer.Next.Cells)[weight_index].ErrorDelta)
			}
			//fmt.Println(cell.Name, "WEIIGHT MULTIPLATION", weight_multiplation)
			error_delta = error_delta * weight_multiplation
			cell.ErrorDelta = error_delta
			//fmt.Println(cell.Name, "ERROR DELTA ISx", cell.ErrorDelta)
		}
	}
}

func (mlp *MLP) CalculateNewWeights() {

	var pointerLayer *Layer
	for i := len(*mlp.Layers) - 1; i >= 0; i-- {
		pointerLayer = &(*mlp.Layers)[i]

		var pointerCell *Cell
		for cell_index := range *pointerLayer.Cells {
			pointerCell = &(*pointerLayer.Cells)[cell_index]
			if pointerCell.OutsideWeights == nil || len(*pointerCell.OutsideWeights) == 0 {
				//fmt.Println("IGNORING CELL", pointerCell.Name)
				continue
			}

			var pointerWeight *CellWeight
			for weight_index := range *pointerCell.OutsideWeights {
				pointerWeight = &(*pointerCell.OutsideWeights)[weight_index]

				var delta float64 = 0
				//fmt.Println("CHECK", pointerCell.Name, pointerCell.Value)
				delta = mlp.LearningRate * pointerCell.Value * (*pointerLayer.Next.Cells)[weight_index].ErrorDelta
				pointerWeight.Value = pointerWeight.Value + delta
				var temp_limit float64
				temp_limit = 0.0000000000000
				if mlp.IterationCount%100 == 0 && delta < temp_limit && false {
					fmt.Println("->>>>>>>")
					fmt.Println(pointerCell.Name, "Learning Rate:", mlp.LearningRate)
					fmt.Println("Pointer Cell Value:", pointerCell.Value)
					fmt.Println("Next Cell Error Delta:", (*pointerLayer.Next.Cells)[weight_index].ErrorDelta)
					fmt.Println("DELTA:", delta, "SUBSTRACT:", delta-temp_limit)
					fmt.Println("-<<<<<<<")
					//fmt.Println( "WEIGHT:", weight_index, "DELTA:", delta, "ACTUAL VALUE:", pointerWeight.Value)
				}
			}
		}
	}
}

func (mlp *MLP) ShuffleInputs() {
	rand.Shuffle(len(*mlp.Inputs), func(i, j int) {
		temp := (*mlp.Inputs)[i]
		(*mlp.Inputs)[i] = (*mlp.Inputs)[j]
		(*mlp.Inputs)[j] = temp
	})
}
