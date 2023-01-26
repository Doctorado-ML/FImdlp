#include "Factorize.h"

namespace utils {
    vector<int> cppFactorize(const vector<string>& labels_t)
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
}