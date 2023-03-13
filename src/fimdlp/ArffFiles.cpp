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
vector<pair<string, string>> ArffFiles::getAttributes()
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
void ArffFiles::load(string fileName, bool classLast)
{
    ifstream file(fileName);
    if (file.is_open()) {
        string line, keyword, attribute, type;
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line == "\r" || line == " ") {
                continue;
            }
            if (line.find("@attribute") != string::npos || line.find("@ATTRIBUTE") != string::npos) {
                stringstream ss(line);
                ss >> keyword >> attribute >> type;
                attributes.push_back({ attribute, type });
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
        if (classLast) {
            className = get<0>(attributes.back());
            classType = get<1>(attributes.back());
            attributes.pop_back();
        } else {
            className = get<0>(attributes.front());
            classType = get<1>(attributes.front());
            attributes.erase(attributes.begin());
        }
        generateDataset(classLast);
    } else
        throw invalid_argument("Unable to open file");
}
void ArffFiles::generateDataset(bool classLast)
{
    X = vector<vector<float>>(attributes.size(), vector<float>(lines.size()));
    vector<string> yy = vector<string>(lines.size(), "");
    int labelIndex = classLast ? attributes.size() : 0;
    for (size_t i = 0; i < lines.size(); i++) {
        stringstream ss(lines[i]);
        string value;
        int pos = 0, xIndex = 0;
        while (getline(ss, value, ',')) {
            if (pos++ == labelIndex) {
                yy[i] = value;
            } else {
                X[xIndex++][i] = stof(value);
            }
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
vector<int> ArffFiles::factorize(const vector<string>& labels_t)
{
    vector<int> yy;
    yy.reserve(labels_t.size());
    map<string, int> labelMap;
    int i = 0;
    for (string label : labels_t) {
        if (labelMap.find(label) == labelMap.end()) {
            labelMap[label] = i++;
        }
        yy.push_back(labelMap[label]);
    }
    return yy;
}