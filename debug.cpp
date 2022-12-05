std::cout << "+++++++++++++++++++++++" << std::endl;
for (size_t i = 0; i < y.size(); i++) {
    printf("(%3.1f, %d)\n", X[indices.at(i)], y[indices.at(i)]);
}
std::cout << "+++++++++++++++++++++++" << std::endl;

std::cout << "Information Gain:" << std::endl;
auto nc = Metrics::numClasses(y, indices, 0, indices.size());
for (auto cutPoint = cutIdx.begin(); cutPoint != cutIdx.end(); ++cutPoint) {
    std::cout << *cutPoint << " -> " << Metrics::informationGain(y, indices, 0, indices.size(), *cutPoint, nc) << std::endl;
    //  << Metrics::informationGain(y, 0, y.size(), *cutPoint, Metrics::numClasses(y, 0, y.size())) << std::endl;
}

def test(self) :
    print("Calculating cut points in python for first feature")
    yz = self.y_.copy()
    xz = X[:, 0].copy()
    xz = xz[np.argsort(X[:, 0])]
    yz = yz[np.argsort(X[:, 0])]
    cuts = []
    for i in range(1, len(yz)) :
        if yz[i] != yz[i - 1] and xz[i - 1] < xz[i] :
            print(f"Cut point: ({xz[i-1]}, {xz[i]}) ({yz[i-1]}, {yz[i]})")
            cuts.append((xz[i] + xz[i - 1]) / 2)
            print("Cuts calculados en python: ", cuts)
            print("-- Cuts calculados en C++ --")
            print("Cut points for each feature in Iris dataset:")
            for i in range(0, 1) :
                # datax = self.X_[np.argsort(self.X_[:, i]), i]
                # y_ = self.y_[np.argsort(self.X_[:, i])]
                datax = self.X_[:, i]
                y_ = self.y_
                self.discretizer_.fit(datax, y_)
                Xcutpoints = self.discretizer_.get_cut_points()
                print(
                    f"New ({len(Xcutpoints)}):{self.features_[i]:20s}: "
                    f"{[i['toValue'] for i in Xcutpoints]}"
                )
                X_translated = [
                    f"{i['classNumber']} - ({i['start']}, {i['end']}) - "
                        f"({i['fromValue']}, {i['toValue']})"
                        for i in Xcutpoints
                ]
                print(X_translated)
                            print("*******************************")
                            print("Disretized values:")
                            print(self.discretizer_.get_discretized_values())
                            print("*******************************")
                            return X