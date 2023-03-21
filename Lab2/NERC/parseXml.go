package main

import (
	"bufio"
	"encoding/xml"
	"fmt"
	"io/ioutil"
	"os"
)

// our struct which contains the complete
// array of all Users in the file
type Drugs struct {
	XMLName xml.Name `xml:"drugbank"`
	Drugs   []Drug   `xml:"drug"`
}

// the user struct, this contains our
// Type attribute, our user's name and
// a social struct which will contain all
// our social links
type Drug struct {
	XMLName xml.Name `xml:"drug"`
	Name    string   `xml:"name"`
}

// a simple struct which contains all our
// social links
type Product struct {
	XMLName xml.Name `xml:"product"`
	Name    string   `xml:"name"`
}

// writeLines writes the lines to the given file.
func writeLines(lines []string, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	w := bufio.NewWriter(file)
	for _, line := range lines {
		fmt.Fprintln(w, line)
	}
	return w.Flush()
}

func main() {

	// Open our xmlFile
	xmlFile, err := os.Open("full_database.xml")
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("Successfully Opened full_database.xml")
	// defer the closing of our xmlFile so that we can parse it later on
	defer xmlFile.Close()

	// read our opened xmlFile as a byte array.
	byteValue, _ := ioutil.ReadAll(xmlFile)

	// we initialize our Users array
	var drugs Drugs
	// we unmarshal our byteArray which contains our
	// xmlFiles content into 'users' which we defined above
	xml.Unmarshal(byteValue, &drugs)

	// we iterate through every user within our users array and
	// print out the user Type, their name, and their facebook url
	// as just an example
	drugNames := make([]string, 0)

	for i := 0; i < len(drugs.Drugs); i++ {
		// fmt.Println("Drugs Name: " + drugs.Drugs[i].Name)
		drugNames = append(drugNames, drugs.Drugs[i].Name)
	}

	if err := writeLines(drugNames, "drugNames.txt"); err != nil {
		fmt.Println("writeLines: %s", err)
	}
}
