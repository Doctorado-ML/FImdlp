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

def test(self):
    print("Calculating cut points in python for first feature")
    yz = self.y_.copy()
    xz = X[:, 0].copy()
    xz = xz[np.argsort(X[:, 0])]
    yz = yz[np.argsort(X[:, 0])]
    cuts = []
    for i in range(1, len(yz)):
        if yz[i] != yz[i - 1] and xz[i - 1] < xz[i] :
            print(f"Cut point: ({xz[i-1]}, {xz[i]}) ({yz[i-1]}, {yz[i]})")
            cuts.append((xz[i] + xz[i - 1]) / 2)
            print("Cuts calculados en python: ", cuts)
            print("-- Cuts calculados en C++ --")
            print("Cut points for each feature in Iris dataset:")
            for i in range(0, 1):
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

                            c++
                            i: 0 4.3, 0
                            i : 1 4.4, 0
                            i : 2 4.4, 0
                            i : 3 4.4, 0
                            i : 4 4.5, 0
                            i : 5 4.6, 0
                            i : 6 4.6, 0
                            i : 7 4.6, 0
                            i : 8 4.6, 0
                            i : 9 4.7, 0
                            i : 10 4.7, 0
                            i : 11 4.8, 0
                            i : 12 4.8, 0
                            i : 13 4.8, 0
                            i : 14 4.8, 0
                            i : 15 4.8, 0
                            i : 16 4.9, 0
                            i : 17 4.9, 0
                            i : 18 4.9, 0
                            i : 19 4.9, 0
                            i : 20 4.9, 1

                            python
                            i : 0 4.3 0
                            i : 1 4.4 0
                            i : 2 4.4 0
                            i : 3 4.4 0
                            i : 4 4.5 0
                            i : 5 4.6 0
                            i : 6 4.6 0
                            i : 7 4.6 0
                            i : 8 4.6 0
                            i : 9 4.7 0
                            i : 10 4.7 0
                            i : 11 4.8 0
                            i : 12 4.8 0
                            i : 13 4.8 0
                            i : 14 4.8 0
                            i : 15 4.8 0
                            i : 16 4.9 1
                            i : 17 4.9 2
                            i : 18 4.9 0
                            i : 19 4.9 0
                            i : 20 4.9 0



                        idx: 20 entropy_left : 0 entropy_right : 0.488187 -> 0 150
                            idx : 21 entropy_left : 0.0670374 entropy_right : 0.489381 -> 0 150
                            idx : 22 entropy_left : 0.125003 entropy_right : 0.490573 -> 0 150
                            idx : 24 entropy_left : 0.11507 entropy_right : 0.482206 -> 0 150
                            idx : 25 entropy_left : 0.162294 entropy_right : 0.483488 -> 0 150
                            idx : 29 entropy_left : 0.141244 entropy_right : 0.462922 -> 0 150
                            idx : 30 entropy_left : 0.178924 entropy_right : 0.464386 -> 0 150
                            idx : 33 entropy_left : 0.163818 entropy_right : 0.444778 -> 0 150
                            idx : 34 entropy_left : 0.195735 entropy_right : 0.44637 -> 0 150
                            idx : 44 entropy_left : 0.154253 entropy_right : 0.339183 -> 0 150
                            idx : 45 entropy_left : 0.178924 entropy_right : 0.34098 -> 0 150
                            idx : 51 entropy_left : 0.159328 entropy_right : 0.217547 -> 0 150
                            idx : 52 entropy_left : 0.180508 entropy_right : 0.219019 -> 0 150
                            idx : 53 entropy_left : 0.177368 entropy_right : 0.189687 -> 0 150
                            idx : 58 entropy_left : 0.265229 entropy_right : 0.196677 -> 0 150
                            idx : 59 entropy_left : 0.261331 entropy_right : 0.162291 -> 0 150
                            idx : 61 entropy_left : 0.289819 entropy_right : 0.164857 -> 0 150
                            idx : 62 entropy_left : 0.302928 entropy_right : 0.166175 -> 0 150
                            idx : 68 entropy_left : 0.36831 entropy_right : 0.174607 -> 0 150
                            idx : 69 entropy_left : 0.364217 entropy_right : 0.131848 -> 0 150
                            idx : 70 entropy_left : 0.373248 entropy_right : 0.133048 -> 0 150
                            idx : 71 entropy_left : 0.381826 entropy_right : 0.134273 -> 0 150
                            idx : 72 entropy_left : 0.377855 entropy_right : 0.0805821 -> 0 150
                            idx : 74 entropy_left : 0.393817 entropy_right : 0.0822096 -> 0 150
                            idx : 75 entropy_left : 0.401218 entropy_right : 0.0830509 -> 0 150
                            idx : 76 entropy_left : 0.397415 entropy_right : 0 -> 0 150
                            idx : 77 entropy_left : 0.4045 entropy_right : 0 -> 0 150
                            idx : 78 entropy_left : 0.411247 entropy_right : 0 -> 0 150
                            idx : 79 entropy_left : 0.417674 entropy_right : 0 -> 0 150
                            idx : 81 entropy_left : 0.429626 entropy_right : 0 -> 0 150
                            idx : 83 entropy_left : 0.440472 entropy_right : 0 -> 0 150
                            idx : 84 entropy_left : 0.445513 entropy_right : 0 -> 0 150
                            idx : 87 entropy_left : 0.459246 entropy_right : 0 -> 0 150
                            idx : 88 entropy_left : 0.463395 entropy_right : 0 -> 0 150
                            idx : 89 entropy_left : 0.467347 entropy_right : 0 -> 0 150
                            idx : 91 entropy_left : 0.474691 entropy_right : 0 -> 0 150
                            idx : 95 entropy_left : 0.487368 entropy_right : 0 -> 0 150
                            idx : 97 entropy_left : 0.492813 entropy_right : 0 -> 0 150
                            idx : 99 entropy_left : 0.497728 entropy_right : 0 -> 0 150
                            idx : 101 entropy_left : 0.502156 entropy_right : 0 -> 0 150
                            idx : 102 entropy_left : 0.504201 entropy_right : 0 -> 0 150
                            idx : 104 entropy_left : 0.507973 entropy_right : 0 -> 0 150
                            idx : 105 entropy_left : 0.509709 entropy_right : 0 -> 0 150
                            idx : 106 entropy_left : 0.511351 entropy_right : 0 -> 0 150
                            idx : 107 entropy_left : 0.512902 entropy_right : 0 -> 0 150
                            idx : 109 entropy_left : 0.515747 entropy_right : 0 -> 0 150
                            idx : 110 entropy_left : 0.517047 entropy_right : 0 -> 0 150
                            idx : 113 entropy_left : 0.520497 entropy_right : 0 -> 0 150
                            idx : 114 entropy_left : 0.521506 entropy_right : 0 -> 0 150
                            idx : 117 entropy_left : 0.524149 entropy_right : 0 -> 0 150
                            idx : 118 entropy_left : 0.52491 entropy_right : 0 -> 0 150
                            idx : 120 entropy_left : 0.526264 entropy_right : 0 -> 0 150
                            idx : 122 entropy_left : 0.52741 entropy_right : 0 -> 0 150
                            idx : 127 entropy_left : 0.52946 entropy_right : 0 -> 0 150
                            idx : 130 entropy_left : 0.530197 entropy_right : 0 -> 0 150
                            idx : 132 entropy_left : 0.530507 entropy_right : 0 -> 0 150
                            idx : 133 entropy_left : 0.530611 entropy_right : 0 -> 0 150
                            idx : 134 entropy_left : 0.530684 entropy_right : 0 -> 0 150
                            idx : 135 entropy_left : 0.530726 entropy_right : 0 -> 0 150
                            idx : 137 entropy_left : 0.530721 entropy_right : 0 -> 0 150
                            idx : 138 entropy_left : 0.530677 entropy_right : 0 -> 0 150
                            cut : 5.5 index : 53
                            start : 0 cut : 53 end : 150
                            k = 3 k1 = 3 k2 = 3 ent = 0.528321 ent1 = 0.177368 ent2 = 0.189687
                            ig = 0.342987 delta = 4.16006 N 150 term 0.0758615
                            ¡Ding!5.5 53


                            idx : 20  entropy_left : 0  entropy_right : 1.5485806065228545  ->  0   150
                            idx : 21  entropy_left : 0.2761954276479391  entropy_right : 1.549829505666378  ->  0   150
                            idx : 22  entropy_left : 0.5304060778306042  entropy_right : 1.5511852922535474  ->  0   150
                            idx : 24  entropy_left : 0.4971501836369671  entropy_right : 1.5419822842863982  ->  0   150
                            idx : 25  entropy_left : 0.6395563653739031  entropy_right : 1.5433449229510985  ->  0   150
                            idx : 29  entropy_left : 0.574828144380386  entropy_right : 1.5202013991459298  ->  0   150
                            idx : 30  entropy_left : 0.6746799231474564  entropy_right : 1.521677608876836  ->  0   150
                            idx : 33  entropy_left : 0.6311718053929063  entropy_right : 1.4992098113026513  ->  0   150
                            idx : 34  entropy_left : 0.7085966983474103  entropy_right : 1.5007111828980744  ->  0   150
                            idx : 44  entropy_left : 0.5928251064639408  entropy_right : 1.3764263022492553  ->  0   150
                            idx : 45  entropy_left : 0.6531791627726858  entropy_right : 1.3779796176519241  ->  0   150
                            idx : 51  entropy_left : 0.5990326006132177  entropy_right : 1.2367928607774141  ->  0   150
                            idx : 52  entropy_left : 0.6496096346956632  entropy_right : 1.2377158231343603  ->  0   150
                            idx : 53  entropy_left : 0.6412482850735854  entropy_right : 1.2046986815511866  ->  0   150
                            idx : 58  entropy_left : 0.8211258609270055  entropy_right : 1.2056112071736118  ->  0   150
                            idx : 59  entropy_left : 0.8128223064150747  entropy_right : 1.167065448996099  ->  0   150
                            idx : 61  entropy_left : 0.8623538561746379  entropy_right : 1.1653351793699953  ->  0   150
                            idx : 62  entropy_left : 0.9353028851500502  entropy_right : 1.1687172769890006  ->  0   150
                            idx : 68  entropy_left : 1.031929035599206  entropy_right : 1.1573913563403753  ->  0   150
                            idx : 69  entropy_left : 1.0246284743137688  entropy_right : 1.109500797247481  ->  0   150
                            idx : 70  entropy_left : 1.036186417911213  entropy_right : 1.105866621101474  ->  0   150
                            idx : 71  entropy_left : 1.0895830429620594  entropy_right : 1.1104593064416028  ->  0   150
                            idx : 72  entropy_left : 1.0822273380873693  entropy_right : 1.0511407586429597  ->  0   150
                            idx : 74  entropy_left : 1.1015727511177442  entropy_right : 1.041722068095403  ->  0   150
                            idx : 75  entropy_left : 1.1457749842070042  entropy_right : 1.0462881865460743  ->  0   150
                            idx : 76  entropy_left : 1.1387129726704701  entropy_right : 0.9568886656798212  ->  0   150
                            idx : 77  entropy_left : 1.1468549240968817  entropy_right : 0.9505668528932196  ->  0   150
                            idx : 78  entropy_left : 1.1848333092150132  entropy_right : 0.9544340029249649  ->  0   150
                            idx : 79  entropy_left : 1.1918623939938016  entropy_right : 0.9477073729342066  ->  0   150
                            idx : 81  entropy_left : 1.2548698305334247  entropy_right : 0.9557589912150009  ->  0   150
                            idx : 83  entropy_left : 1.2659342914094807  entropy_right : 0.9411864371816835  ->  0   150
                            idx : 84  entropy_left : 1.2922669208691815  entropy_right : 0.9456603046006402  ->  0   150
                            idx : 87  entropy_left : 1.3041589171425696  entropy_right : 0.9182958340544896  ->  0   150
                            idx : 88  entropy_left : 1.327572716814381  entropy_right : 0.9235785996175947  ->  0   150
                            idx : 89  entropy_left : 1.330465426809402  entropy_right : 0.9127341558073343  ->  0   150
                            idx : 91  entropy_left : 1.3709454625942779  entropy_right : 0.9238422284571814  ->  0   150
                            idx : 95  entropy_left : 1.378063041001916  entropy_right : 0.8698926856041563  ->  0   150
                            idx : 97  entropy_left : 1.4115390027326744  entropy_right : 0.8835850861052532  ->  0   150
                            idx : 99  entropy_left : 1.4130351465796736  entropy_right : 0.8478617451660526  ->  0   150
                            idx : 101  entropy_left : 1.4412464483479606  entropy_right : 0.863120568566631  ->  0   150
                            idx : 102  entropy_left : 1.4415827640191903  entropy_right : 0.8426578772022391  ->  0   150
                            idx : 104  entropy_left : 1.4655411381577925  entropy_right : 0.8589810370425963  ->  0   150
                            idx : 105  entropy_left : 1.465665295753282  entropy_right : 0.8366407419411673  ->  0   150
                            idx : 106  entropy_left : 1.4762911618692924  entropy_right : 0.8453509366224365  ->  0   150
                            idx : 107  entropy_left : 1.4762132849962355  entropy_right : 0.8203636429576732  ->  0   150
                            idx : 109  entropy_left : 1.4951379218217782  entropy_right : 0.8390040613676977  ->  0   150
                            idx : 110  entropy_left : 1.4949188482339508  entropy_right : 0.8112781244591328  ->  0   150
                            idx : 113  entropy_left : 1.5183041104369397  entropy_right : 0.8418521897563207  ->  0   150
                            idx : 114  entropy_left : 1.51802714866133  entropy_right : 0.8112781244591328  ->  0   150
                            idx : 117  entropy_left : 1.5364854516368571  entropy_right : 0.8453509366224365  ->  0   150
                            idx : 118  entropy_left : 1.5361890331151247  entropy_right : 0.8112781244591328  ->  0   150
                            idx : 120  entropy_left : 1.5462566034163763  entropy_right : 0.8366407419411673  ->  0   150
                            idx : 122  entropy_left : 1.545378825051491  entropy_right : 0.74959525725948  ->  0   150
                            idx : 127  entropy_left : 1.5644893588382582  entropy_right : 0.828055725379504  ->  0   150
                            idx : 130  entropy_left : 1.562956340286807  entropy_right : 0.6098403047164004  ->  0   150
                            idx : 132  entropy_left : 1.5687623685201277  entropy_right : 0.6500224216483541  ->  0   150
                            idx : 133  entropy_left : 1.5680951037987416  entropy_right : 0.5225593745369408  ->  0   150
                            idx : 134  entropy_left : 1.5706540443736308  entropy_right : 0.5435644431995964  ->  0   150
                            idx : 135  entropy_left : 1.5699201014782036  entropy_right : 0.35335933502142136  ->  0   150
                            idx : 137  entropy_left : 1.5744201314186457  entropy_right : 0.39124356362925566  ->  0   150
                            idx : 138  entropy_left : 1.5736921054134685  entropy_right : 0  ->  0   150
                            ¡Ding!4.9 20

                            k = 2  k1 = 1  k2 = 2  ent = 0.5225593745369408  ent1 = 0  ent2 = 0.5435644431995964
                            ig = 0.010969310349085326  delta = 2.849365059382915  N  17  term  0.4029038270225244
                            idx : 135  entropy_left : 0  entropy_right : 0.35335933502142136  ->  134   150
                            idx : 137  entropy_left : 0.9182958340544896  entropy_right : 0.39124356362925566  ->  134   150
                            idx : 138  entropy_left : 1.0  entropy_right : 0  ->  134   150
                            start : 134  cut : 135  end : 150
                            k = 2  k1 = 1  k2 = 2  ent = 0.5435644431995964  ent1 = 0  ent2 = 0.35335933502142136
                            ig = 0.21229006661701388  delta = 2.426944705701254  N  16  term  0.39586470633186077
                            idx : 137  entropy_left : 0  entropy_right : 0.39124356362925566  ->  135   150
                            idx : 138  entropy_left : 0.9182958340544896  entropy_right : 0  ->  135   150
                            start : 135  cut : 137  end : 150
                            k = 2  k1 = 1  k2 = 2  ent = 0.35335933502142136  ent1 = 0  ent2 = 0.39124356362925566
                            ig = 0.01428157987606643  delta = 2.8831233792732727  N  15  term  0.44603188675539174
                            idx : 138  entropy_left : 0  entropy_right : 0  ->  137   150
                            start : 137  cut : 138  end : 150
                            k = 2  k1 = 1  k2 = 1  ent = 0.39124356362925566  ent1 = 0  ent2 = 0
                            ig = 0.39124356362925566  delta = 2.0248677947990927  N  13  term  0.4315254073477115
                            [[4.9, 5.2, 5.4, 6.75]]


                    cut : 1.4 index : 81
                        start : 50 cut : 81 end : 96
                        k = 2 k1 = 2 k2 = 1 ent = 0.151097 ent1 = 0.205593 ent2 = 0
                        ig = 0.0125455 delta = 2.91635 N 46 term 0.182787
                        idx : 80 entropy_left : 0 entropy_right : 0 -> 50 81
                        cut : 1.4 index : 80
                        start : 50 cut : 80 end : 81
                        k = 2 k1 = 1 k2 = 1 ent = 0.205593 ent1 = 0 ent2 = 0
                        ig = 0.205593 delta = 2.39617 N 31 term 0.235583
                        idx : 112 entropy_left : 0 entropy_right : 0.175565 -> 103 150
                        idx : 113 entropy_left : 0.468996 entropy_right : 0 -> 103 150
                        cut : 1.8 index : 112
                        start : 103 cut : 112 end : 150
                        k = 2 k1 = 1 k2 = 2 ent = 0.148549 ent1 = 0 ent2 = 0.175565
                        ig = 0.00660326 delta = 2.86139 N 47 term 0.178403
                        idx : 113 entropy_left : 0 entropy_right : 0 -> 112 150
                        cut : 1.8 index : 113
                        start : 112 cut : 113 end : 150
                        k = 2 k1 = 1 k2 = 1 ent = 0.175565 ent1 = 0 ent2 = 0
                        ig = 0.175565 delta = 2.45622 N 38 term 0.201728
                        [[4.900000095367432, 4.949999809265137, 5.0, 5.099999904632568, 5.199999809265137, 5.25, 5.400000095367432, 5.449999809265137,
                        5.5, 5.550000190734863, 5.599999904632568, 5.699999809265137, 5.800000190734863, 5.900000095367432, 5.949999809265137, 6.0, 6.050000190734863,
                        6.099999904632568, 6.149999618530273, 6.199999809265137, 6.25, 6.300000190734863, 6.400000095367432, 6.5, 6.550000190734863, 6.649999618530273, 6.699999809265137,
                        6.75, 6.800000190734863, 6.850000381469727, 6.900000095367432, 6.949999809265137, 7.050000190734863]]