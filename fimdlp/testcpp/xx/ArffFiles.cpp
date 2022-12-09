#include "ArffFiles.h"

#include <fstream>
#include <sstream>
#include <map>
#include <iostream>

using namespace std;

ArffFiles::ArffFiles()
{
}
vector<string> ArffFiles::getLines()
{
    return lines;
}
unsigned long int ArffFiles::getSize()
{
    return lines.size();
}
vector<tuple<string, string>> ArffFiles::getAttributes()
{
    return attributes;
}
string ArffFiles::getClassName()
{
    return className;
}
string ArffFiles::getClassType()
{
    return classType;
}
vector<vector<float>>& ArffFiles::getX()
{
    return X;
}
vector<int>& ArffFiles::getY()
{
    return y;
}
void ArffFiles::load(string fileName)
{
    ifstream file(fileName);
    string keyword, attribute, type;
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            if (line[0] == '%' || line.empty() || line == "\r" || line == " ") {
                continue;
            }
            if (line.find("@attribute") != string::npos || line.find("@ATTRIBUTE") != string::npos) {
                stringstream ss(line);
                ss >> keyword >> attribute >> type;
                attributes.push_back(make_tuple(attribute, type));
                continue;
            }
            if (line[0] == '@') {
                continue;
            }
            lines.push_back(line);
        }
        file.close();
        if (attributes.empty())
            throw invalid_argument("No attributes found");
        className = get<0>(attributes.back());
        classType = get<1>(attributes.back());
        attributes.pop_back();
        generateDataset();
    } else
        throw invalid_argument("Unable to open file");
}
void ArffFiles::generateDataset()
{
    X = vector<vector<float>>(lines.size(), vector<float>(attributes.size()));
    vector<string> yy = vector<string>(lines.size(), "");
    for (int i = 0; i < lines.size(); i++) {
        stringstream ss(lines[i]);
        string value;
        int j = 0;
        while (getline(ss, value, ',')) {
            if (j == attributes.size()) {
                yy[i] = value;
                break;
            }
            X[i][j] = stof(value);
            j++;
        }
    }
    y = factorize(yy);
}
string ArffFiles::trim(const string& source)
{
    string s(source);
    s.erase(0, s.find_first_not_of(" \n\r\t"));
    s.erase(s.find_last_not_of(" \n\r\t") + 1);
    return s;
}
vector<int> ArffFiles::factorize(const vector<string>& labels)
{
    vector<int> yy;
    yy.reserve(labels.size());
    map<string, int> labelMap;
    int i = 0;
    for (string label : labels) {
        if (labelMap.find(label) == labelMap.end()) {
            labelMap[label] = i++;
        }
        yy.push_back(labelMap[label]);
    }
    return yy;
}