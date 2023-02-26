#ifndef ARFFFILES_H
#define ARFFFILES_H
#include <string>
#include <vector>
using namespace std;
class ArffFiles {
private:
    vector<string> lines;
    vector<pair<string, string>> attributes;
    string className, classType;
    vector<vector<float>> X;
    vector<int> y;
    void generateDataset(bool);
public:
    ArffFiles();
    void load(string, bool = true);
    vector<string> getLines();
    unsigned long int getSize();
    string getClassName();
    string getClassType();
    string trim(const string&);
    vector<vector<float>>& getX();
    vector<int>& getY();
    vector<pair<string, string>> getAttributes();
    vector<int> factorize(const vector<string>& labels_t);
};
#endif