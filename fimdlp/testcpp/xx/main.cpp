#include "ArffFiles.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

int main(int argc, char **argv) {
    ArffFiles file;
    vector<string> lines;
    //file.load("datasets/mfeat-factors.arff");
    file.load("/Users/rmontanana/Code/FImdlp/fimdlp/testcpp/datasets/mfeat-factors.arff");
    cout << "Number of lines: " << file.getSize() << endl;
    cout << "Attributes: " << endl;
    for (auto attribute: file.getAttributes()) {
        cout << "Name: " << get<0>(attribute) << " Type: " << get<1>(attribute) << endl;
    }
    cout << "Class name: " << file.getClassName() << endl;
    cout << "Class type: " << file.getClassType() << endl;
    cout << "Data: " << endl;
    vector<vector<float>> &X = file.getX();
    vector<int> &y = file.getY();
    for (int i = 0; i < X.size(); i++) {
        for (float value: X[i]) {
            cout << fixed << setprecision(1) << value << " ";
        }
        cout << y[i] << endl;
    }
    return 0;
}
