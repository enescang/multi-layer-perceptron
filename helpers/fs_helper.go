package helpers

import "os"

func ReadFile(path string) (string, error) {
	fileExists := IsFileExists(path)
	if fileExists == false {
		return "", os.ErrNotExist
	}
	fileByteArr, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(fileByteArr), nil
}

func WriteToFile(path string, data []byte) error {
	err := os.WriteFile(path, data, 0644)
	return err
}

func IsFileExists(path string) bool {
	fileStat, err := os.Stat(path)
	if err != nil {
		return false
	}
	return fileStat.IsDir() == false
}
