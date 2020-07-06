#include <iostream>
#include <fstream>

#include "UKF.h"

int main(void)
{
    UKF ukf{4};

    uint32_t n = 1000;

    std::vector<float> x_measure;
    std::vector<float> y_measure;

    std::ifstream data_in{ "data.txt" };
    for (uint32_t i = 0; i < n; i++)
    {
        std::string line;
        std::string::size_type sz;     // alias of size_t

        std::getline(data_in, line);
        x_measure.push_back(std::stof(line, &sz));
        y_measure.push_back(std::stof(line.substr(sz)));

    }
    data_in.close();

    std::vector<float> x_pred;
    std::vector<float> y_pred;

    // init the state of the ukf
    ukf.state(0) = x_measure[0];
    ukf.state(1) = y_measure[0];

    for (uint32_t i = 0; i < n; i++)
    {
        ukf.predict(0.001);
        Eigen::VectorXf measure{ 2 };
        measure(0) = x_measure[i];
        measure(1) = y_measure[i];

        ukf.update(measure);
        
        x_pred.push_back(ukf.state(0));
        y_pred.push_back(ukf.state(1));
    }

    std::ofstream out{ "data_out.txt" };

    for (uint32_t i = 0; i < n; i++)
    {
        out << x_pred[i] << " " << y_pred[i] << "\n";
    }
    out.close();

    std::cout << "done\n";
}
