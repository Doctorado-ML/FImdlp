#include "ArffFiles.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include "../CPPFImdlp.h"

using namespace std;

int main(int argc, char** argv)
{
    ArffFiles file;
    vector<string> lines;
    string path = "../../tests/datasets/";
    map<string, bool > datasets = {
        {"mfeat-factors", true},
        {"iris", true},
        {"letter", true},
        {"kdd_JapaneseVowels", false}
    };
    if (argc != 2 || datasets.find(argv[1]) == datasets.end()) {
        cout << "Usage: " << argv[0] << " {mfeat-factors, iris, letter, kdd_JapaneseVowels}" << endl;
        return 1;
    }

    file.load(path + argv[1] + ".arff", datasets[argv[1]]);
    auto attributes = file.getAttributes();
    int items = file.getSize();
    cout << "Number of lines: " << items << endl;
    cout << "Attributes: " << endl;
    for (auto attribute : attributes) {
        cout << "Name: " << get<0>(attribute) << " Type: " << get<1>(attribute) << endl;
    }
    cout << "Class name: " << file.getClassName() << endl;
    cout << "Class type: " << file.getClassType() << endl;
    cout << "Data: " << endl;
    vector<vector<float>>& X = file.getX();
    vector<int>& y = file.getY();
    for (int i = 0; i < 50; i++) {
        for (auto feature : X) {
            cout << fixed << setprecision(1) << feature[i] << " ";
        }
        cout << y[i] << endl;
    }
    mdlp::CPPFImdlp test = mdlp::CPPFImdlp(false);
    for (auto i = 0; i < attributes.size(); i++) {
        cout << "Cut points for " << get<0>(attributes[i]) << endl;
        cout << "--------------------------" << setprecision(3) << endl;
        test.fit(X[i], y);
        for (auto item : test.getCutPoints()) {
            cout << item << endl;
        }
    }
    return 0;
}
