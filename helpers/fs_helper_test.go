package helpers

import (
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"
)

var existsFilePath string = "./file_exists.txt"
var nonExistsFilePath string = "./not_exists.json"

func TestFileExists(t *testing.T) {
	isExists := IsFileExists(existsFilePath)
	if isExists == false {
		t.Errorf("IsFileExists says the file is NOT exists but it is there")
	}
}
func TestFileNotExists(t *testing.T) {
	//Need to be sure file removed :D
	os.Remove(nonExistsFilePath)
	isExists := IsFileExists(nonExistsFilePath)
	if isExists {
		t.Errorf("IsFileExists says the file is exists but it is NOT there")
	}
}

func TestWriteAndReadFile(t *testing.T) {
	rand.Seed(time.Now().UnixMicro())
	randomText := strconv.FormatFloat(rand.Float64(), 'E', -1, 64)
	WriteToFile(existsFilePath, []byte(randomText))
	fileData, err := ReadFile(existsFilePath)
	if err != nil {
		t.Errorf(err.Error())
	}
	if randomText != fileData {
		t.Error("The text that has been written to file is NOT same with the text loaded from file. Expected:", randomText, "Current:", fileData)
	}
}
