#ifndef ARFFFILES_H
#define ARFFFILES_H

#include <string>
#include <vector>

using namespace std;

class ArffFiles {
private:
    vector<string> lines;
    vector<pair<string, string>> attributes;
    string className;
    string classType;
    vector<vector<float>> X;
    vector<int> y;

    void generateDataset(bool);

public:
    ArffFiles();

    void load(const string &, bool = true);

    vector<string> getLines() const;

    unsigned long int getSize() const;

    string getClassName() const;

    string getClassType() const;

    static string trim(const string &);

    vector<vector<float>> &getX();

    vector<int> &getY();

    vector<pair<string, string>> getAttributes() const;

    static vector<int> factorize(const vector<string> &labels_t);
};

#endif