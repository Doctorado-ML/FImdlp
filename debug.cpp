std::cout << "+++++++++++++++++++++++" << std::endl;
for (size_t i = 0; i < y.size(); i++)
{
    printf("(%3.1f, %d)\n", X[indices.at(i)], y[indices.at(i)]);
}
std::cout << "+++++++++++++++++++++++" << std::endl;

std::cout << "Information Gain:" << std::endl;
auto nc = Metrics::numClasses(y, indices, 0, indices.size());
for (auto cutPoint = cutIdx.begin(); cutPoint != cutIdx.end(); ++cutPoint)
{
    std::cout << *cutPoint << " -> " << Metrics::informationGain(y, indices, 0, indices.size(), *cutPoint, nc) << std::endl;
    //  << Metrics::informationGain(y, 0, y.size(), *cutPoint, Metrics::numClasses(y, 0, y.size())) << std::endl;
}