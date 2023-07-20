package mlp

type Layer struct {
	Name     string
	Prev     *Layer
	Next     *Layer
	Cells    *[]Cell
	IsInput  bool
	IsOutput bool
}

func (layer *Layer) AddCell(cell Cell) {
	if layer.Cells == nil {
		layer.Cells = &[]Cell{}
	}
	*layer.Cells = append(*layer.Cells, cell)
}
