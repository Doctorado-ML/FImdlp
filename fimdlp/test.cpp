#include "CPPFImdlp.h"
#include <iostream>

using namespace std;
int main(int argc, char *argv[], char *envp[])
{
    {
        CPPFImdlp::CPPFImdlp fimdlp = CPPFImdlp::CPPFImdlp(true);
        vector<float> X = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        vector<int> y = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        vector<float> cutPts = fimdlp.cutPoints(X, y);
        for (auto &cutPt : cutPts)
        {
            cout << cutPt << endl;
        }
        return 0;
    }
}