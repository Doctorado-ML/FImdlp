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
    size_t col;
    vector<string> row;
    string line, word;
    vector<vector<float>> dataset = vector<vector<float>>(15, vector<float>());
    while (getline(fin, line)) {
        if (count++ > 215) {
            stringstream ss(line);
            col = 0;
            while (getline(ss, word, ',')) {
                col = col % 15;
                dataset[col].push_back(stof(word));
                cout << col << "-" << word << " ";
                col++;
            }
            cout << endl;
        }
    }
    labels y = labels(dataset[0].begin(), dataset[0].end());
    cout << "Column 0 (y): " << y.size() << endl;
    for (auto item : y) {
        cout << item << " ";
    }
    CPPFImdlp test = CPPFImdlp(false, 6, true);
    test.fit(dataset[3], y);
    cout << "Cut points: " << test.getCutPoints().size() << endl;
    for (auto item : test.getCutPoints()) {
        cout << item << " ";
    }
    fin.close();
    return 0;
}