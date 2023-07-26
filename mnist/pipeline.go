package mnist

//File to Pixel struct: https://stackoverflow.com/a/41185404

import (
	"fmt"
	"image"
	"io"
	"io/fs"
	"io/ioutil"
	"os"

	"github.com/enescang/multi-layer-perceptron/mlp"
)

type Pipeline struct {
	MLP    *mlp.MLP
	Path   string
	Batchs []FileBatch
}

type FileBatch struct {
	Path      string
	Files     []fs.FileInfo
	Expecteds []float64
}

//Read all files based on path
//Extract pixels of image file
//Convert Pixels to 0-1 white/black only (RGBA(0,0,0,255) => black)
//Return mlp.InputRow for each file

func (pipeline *Pipeline) AppendToFiles(folder_name string, expecteds []float64) {
	var batch FileBatch
	path := pipeline.Path + "/" + folder_name
	batch.Path = path

	files, err := ioutil.ReadDir(path)
	if err != nil {
		fmt.Println("ReadAllFiles error:", err)
	}
	batch.Files = files
	batch.Expecteds = expecteds
	pipeline.Batchs = append(pipeline.Batchs, batch)
}

func (pipeline *Pipeline) PushFilesToMLPAsInput() {

	for _, batch := range pipeline.Batchs {
		for i := 0; i < 1; i++ {
			fileInfo := batch.Files[i]
			inputRow, err := pipeline.FileInfoToInputRow(batch, fileInfo)
			if err != nil {
				fmt.Println("MINST Extract Pixels Error", err)
				break
			}
			*pipeline.MLP.Inputs = append(*pipeline.MLP.Inputs, *inputRow)
		}
	}
}

func (pipeline *Pipeline) FileInfoToInputRow(batch FileBatch, fileInfo fs.FileInfo) (*mlp.InputRow, error) {
	path := batch.Path + "/" + fileInfo.Name()
	file, err := os.Open(path)
	if err != nil {
		fmt.Println("OPEN FILE ERROR:", err, "path:", file)
		return nil, err
	}
	defer file.Close()

	pixels, err := getPixels(file)
	if err != nil {
		fmt.Println("Get pixel error", err)
		return nil, err
	}
	singleDimPixel := multiDimesionPixelToSingleDimension(pixels)
	inputRow := singleDimensionPixelToInputRow(singleDimPixel)
	inputRow.Expecteds = batch.Expecteds
	return &inputRow, nil
}

// Get the bi-dimensional pixel array
func getPixels(file io.Reader) ([][]Pixel, error) {
	img, _, err := image.Decode(file)

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var pixels [][]Pixel
	for y := 0; y < height; y++ {
		var row []Pixel
		for x := 0; x < width; x++ {
			//a, b, c, d := img.At(x, y).RGBA()
			//fmt.Println(x, y, a, b, c, d)
			row = append(row, rgbaToPixel(img.At(x, y).RGBA()))
		}
		pixels = append(pixels, row)
	}

	return pixels, nil
}

// img.At(x, y).RGBA() returns four uint32 values; we want a Pixel
func rgbaToPixel(r uint32, g uint32, b uint32, a uint32) Pixel {
	R := int(r / 257)
	G := int(g / 257)
	B := int(b / 257)
	A := int(a / 257)
	return Pixel{R, G, B, A, (R == 0 && B == 0 && G == 0)}
}

func multiDimesionPixelToSingleDimension(pixel [][]Pixel) []Pixel {
	var dim []Pixel
	for y := 0; y < len(pixel); y++ {

		for x := 0; x < len(pixel[0]); x++ {
			/*pp := pixel[y][x]
			if pp.R == 0 && pp.G == 0 && pp.B == 0 {
				//fmt.Print("@ ")
				pixel[y][x].IsBlack = true
			} else {
				pixel[y][x].IsBlack = false
				//fmt.Print("- ")
			}*/
			dim = append(dim, pixel[y][x])
		}
	}

	return dim
}

func singleDimensionPixelToInputRow(pixel []Pixel) mlp.InputRow {
	var inputRow mlp.InputRow
	for _, v := range pixel {
		var value float64
		if v.IsBlack {
			value = 0
		} else {
			value = 1
		}
		inputRow.Inputs = append(inputRow.Inputs, value)
	}
	return inputRow
}

func PrintPixel(row mlp.InputRow) {
	for index, v := range row.Inputs {
		if v == 0 {
			fmt.Print("@")
		} else {
			fmt.Print("-")
		}
		if index%28 == 0 {
			fmt.Println()
		}
	}
}
