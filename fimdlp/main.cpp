#include "CPPFImdlp.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;
using namespace mdlp;

int main()
{
    ifstream fin("kdd_JapaneseVowels.arff");
    if (!fin.is_open()) {
        cout << "Error opening file" << endl;
        return 1;
    }

    int count = 0;

    // Read the Data from the file
    // as String Vector
    vector<string> row;
    string line, word;
    while (getline(fin, line)) {
        if (count++ > 215) {
            row.clear();
            stringstream ss(line);
            while (getline(ss, word, ',')) {
                row.push_back(word);
                cout << word << " ";
            }
            cout << endl;
        }
    }
    fin.close();
    return 0;
}