package mnist

import (
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/enescang/multi-layer-perceptron/mlp"
)

// Pixel struct example
type Pixel struct {
	R       int
	G       int
	B       int
	A       int
	IsBlack bool
}

func Run() {
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
	var minstMlp *mlp.MLP
	minstMlp = &mlp.MLP{}
	minstMlp.LearningRate = 0.5
	minstMlp.ErrorRate = 10

	inputLayer := HandleInputLayer(28 * 28)
	hl1 := RandomLayer(len(*(*inputLayer.Cells)[0].OutsideWeights), 2)
	hl2 := RandomLayer(2, 1)
	//hl3 := RandomLayer(3, 1)
	//hl4 := RandomLayer(2, 5)
	//hl5 := RandomLayer(5, 8)
	//hl6 := RandomLayer(8, 9)
	//hl7 := RandomLayer(9, 11)
	//hl8 := RandomLayer(11, 1)

	outputLayer := HandleOutputLayer()

	minstMlp.Inputs = &[]mlp.InputRow{}
	minstMlp.Layers = &[]mlp.Layer{*inputLayer, hl1, hl2, *outputLayer}
	minstMlp.Build()

	var pipeline *Pipeline
	pipeline = &Pipeline{}
	pipeline.Path = "minst_dataset/training"
	pipeline.MLP = minstMlp
	//pipeline.AppendToFiles("0", []float64{0.0})
	pipeline.AppendToFiles("1", []float64{0.1})
	pipeline.AppendToFiles("2", []float64{0.2})
	pipeline.AppendToFiles("3", []float64{0.3})
	pipeline.AppendToFiles("4", []float64{0.4})
	pipeline.AppendToFiles("5", []float64{0.5})
	pipeline.AppendToFiles("6", []float64{0.6})
	pipeline.AppendToFiles("7", []float64{0.7})
	pipeline.AppendToFiles("8", []float64{0.8})
	pipeline.AppendToFiles("9", []float64{0.9})

	pipeline.PushFilesToMLPAsInput()

	fmt.Println("TOTAL INPUT", len(*pipeline.MLP.Inputs))

	start_at := time.Now()
	for {
		completed := minstMlp.Iteration()
		if completed {
			break
		}
	}
	duration := time.Since(start_at)
	fmt.Println("MINST TRAIN DURATION", duration)

	testPipeline := GetTestPipeline()
	testPipeline.AppendToFiles("4", []float64{0.4})
	testPipeline.PushFilesToMLPAsInput()

	testRow, err := testPipeline.FileInfoToInputRow(testPipeline.Batchs[0], testPipeline.Batchs[0].Files[0])
	if err != nil {
		fmt.Println("Test dataset read info error", err, testPipeline.Batchs[0].Files[0])
		return
	}
	result := minstMlp.IterateWithRow(*testRow)
	fmt.Println(result)

	testRow2, err := pipeline.FileInfoToInputRow(pipeline.Batchs[0], pipeline.Batchs[0].Files[0])
	if err != nil {
		fmt.Println("2Test dataset read info error", err)
		return
	}
	result2 := pipeline.MLP.IterateWithRow(*testRow2)
	fmt.Println(result2)
	//PrintPixel((*testPipeline.MLP.Inputs)[0])
}

func Rename(path string) {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		fmt.Println("Rename error:", err)
	}

	for index, file := range files {
		fmt.Println("Index", index)
		old_path := path + "/" + file.Name()
		new_name := path + "/" + strconv.Itoa(index) + ".png"
		fmt.Println(old_path, "->", new_name)
		os.Rename(old_path, new_name)
	}
}

func HandleInputLayer(cell_count int) *mlp.Layer {
	layer1 := mlp.Layer{}
	layer1.IsInput = true
	layer1.Name = "INPUT LAYER"
	layer1.Cells = &[]mlp.Cell{}
	for i := 0; i < cell_count; i++ {
		*layer1.Cells = append(*layer1.Cells, RandomCell(3))
	}
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
			{Value: 0.0000408248},
			{Value: 0.0000408248},
			{Value: 0.0000408248},
		},
	}
	cell2 := mlp.Cell{
		Name:       "Second Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.0000408248},
			{Value: 0.0000408248},
			{Value: 0.0000408248},
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
			{Value: 0.0000408248},
		},
	}
	cell2 := mlp.Cell{
		Name:       "Third Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.0000408248},
		},
	}
	cell3 := mlp.Cell{
		Name:       "Third Layer Cell 2",
		Value:      0,
		ValueDelta: 0,
		OutsideWeights: &[]mlp.CellWeight{
			{Value: 0.0000408248},
		},
	}
	layer1.AddCell(cell1)
	layer1.AddCell(cell2)
	layer1.AddCell(cell3)
	fmt.Println(cell3)
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

func GetTestPipeline() *Pipeline {
	var pipeline *Pipeline
	pipeline = &Pipeline{}

	var test_mlp *mlp.MLP
	test_mlp = &mlp.MLP{}
	test_mlp.Inputs = &[]mlp.InputRow{}

	pipeline.Path = "minst_dataset/testing"
	pipeline.MLP = test_mlp
	return pipeline
}

func RandomLayer(cell_count int, weight_count int) mlp.Layer {
	var layer mlp.Layer
	for i := 0; i < cell_count; i++ {
		layer.AddCell(RandomCell(weight_count))
	}

	return layer
}

func RandomCell(weight_size int) mlp.Cell {
	cell2 := mlp.Cell{
		Name:           "Random Cell",
		Value:          0,
		ValueDelta:     0,
		OutsideWeights: &[]mlp.CellWeight{},
	}
	for i := 0; i < weight_size; i++ {
		*cell2.OutsideWeights = append(*cell2.OutsideWeights, mlp.CellWeight{Value: getCellValue()})
	}
	return cell2
}

func getCellValue() float64 {
	var a float64
	a = math.Sqrt(2 / (784 + 1))
	a = 1*rand.Float64() + rand.Float64() + rand.Float64() + rand.Float64() + rand.Float64()
	if rand.Float64() < 0.3 {
		a = a * -1
	}
	//fmt.Println(a)
	return a
}
