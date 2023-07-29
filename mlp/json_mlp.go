package mlp

import (
	"encoding/json"
	"fmt"
	"time"
)

type JsonMlp struct {
	Name          string         `json:"name"`
	InputRowCount int            `json:"input_row_count"`
	CreatedAt     time.Time      `json:"created_at"`
	Layers        []JsonMlpLayer `json:"layers"`
}

type JsonMlpLayer struct {
	Name     string        `json:"name"`
	IsInput  bool          `json:"is_input"`
	IsOutput bool          `json:"is_output"`
	Cells    []JsonMlpCell `json:"cells"`
}

type JsonMlpCell struct {
	Name    string              `json:"name"`
	Weights []JsonMlpCellWeight `json:"weights"`
}

type JsonMlpCellWeight struct {
	Value float64 `json:"value"`
}

func (mlp *MLP) ToJSON() (string, error) {
	var jsonMlp JsonMlp
	jsonMlp = mlp.toJsonMlPStruct()
	jsonMlp.CreatedAt = time.Now()
	jsonMlp.InputRowCount = len(*mlp.Inputs)
	jsonBytes, err := json.Marshal(jsonMlp)
	if err != nil {
		fmt.Println("mlp.JsonMlpStructToJsonString Error:", err)
		return "", err
	}
	str := string(jsonBytes)
	return str, nil
}

func (mlp *MLP) FromJSON(jsonString string) error {
	jsonMlp, err := mlp.jsonStringToJsonMlpStruct(jsonString)
	if err != nil {
		return err
	}
	mlp.loadLayersFromJsonStruct(*jsonMlp)
	return nil
}

// MLP to JSON string
func (mlp *MLP) toJsonMlPStruct() JsonMlp {
	var jsonMlp JsonMlp
	var jsonMlpLayers []JsonMlpLayer
	for _, layer := range *mlp.Layers {
		var jsonMlpCells []JsonMlpCell
		for _, cell := range *layer.Cells {
			var jsonMlpCellWeights []JsonMlpCellWeight
			for _, weight := range *cell.OutsideWeights {
				jsonMlpCellWeights = append(jsonMlpCellWeights, JsonMlpCellWeight{weight.Value})
			}
			jsonMlpCells = append(jsonMlpCells, JsonMlpCell{Weights: jsonMlpCellWeights, Name: cell.Name})
		}
		jsonMlpLayers = append(jsonMlpLayers, JsonMlpLayer{Name: layer.Name, Cells: jsonMlpCells, IsInput: layer.IsInput, IsOutput: layer.IsOutput})
	}
	jsonMlp.Name = "Sum"
	jsonMlp.Layers = jsonMlpLayers
	return jsonMlp
}

// JSON string to MLP
func (mlp *MLP) jsonStringToJsonMlpStruct(jsonStr string) (*JsonMlp, error) {
	var jsonMlp *JsonMlp
	jsonMlp = &JsonMlp{}

	err := json.Unmarshal([]byte(jsonStr), jsonMlp)
	if err != nil {
		return nil, err
	}
	return jsonMlp, nil
}

func (mlp *MLP) loadLayersFromJsonStruct(jsonMlp JsonMlp) {
	//Reset Layers
	*mlp.Layers = []Layer{}
	for _, layer := range jsonMlp.Layers {
		var cells *[]Cell
		cells = &[]Cell{}
		for _, cell := range layer.Cells {
			var cellWeights *[]CellWeight
			cellWeights = &[]CellWeight{}
			for _, weight := range cell.Weights {
				*cellWeights = append(*cellWeights, CellWeight{Value: weight.Value})
			}
			*cells = append(*cells, Cell{Name: cell.Name, OutsideWeights: cellWeights})
		}
		*mlp.Layers = append(*mlp.Layers, Layer{Cells: cells, Name: layer.Name, IsInput: layer.IsInput, IsOutput: layer.IsOutput})
	}
}
