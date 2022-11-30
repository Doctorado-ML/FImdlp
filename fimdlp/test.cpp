#include "CPPFImdlp.h"
#include <iostream>

using namespace mdlp;
int main(int argc, char *argv[], char *envp[])
{
    {
        CPPFImdlp fimdlp = CPPFImdlp(true);
        std::vector<float> X = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> y = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        fimdlp.computeCutPoints(X, y);
        for (struct CutPointBody cutPt : fimdlp.getCutPoints())
        {
            std::cout << cutPt << std::endl;
        }
        return 0;
    }
}