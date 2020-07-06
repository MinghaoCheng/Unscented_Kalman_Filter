#pragma once

#include <Eigen/dense>
#include <Eigen/Cholesky>

#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

class UKF
{
public:
    UKF(uint8_t n);
    ~UKF();

    VectorXf state;             // [x, y, x_dot, y_dot].transpose
    MatrixXf cov;

    // assumption: constant velocity
    VectorXf model_function(VectorXf input, float dt);

    void predict(float dt);
    void update(VectorXf measure);
    
private:
    // params of the filter
    MatrixXf Q;                         // process noise covariance matrix, v*vT , v ~ N(0, sigma^2)
    float Ax;                           // process noise for augmented acceleration
    float Ay;
    float Px;                           // measurement noise for x axis
    float Py;                           // measurement noise for y axis

    // params of the unscented transform
    uint8_t n, m;                       // m = n + 1, there would be m sigma points
    uint8_t n_aug;
    float lambda;
    VectorXf w;                         // weight of each sigma point
    MatrixXf SigmaPoints;               // In these matrices, data are stored as colmn vectors
    MatrixXf SigmaPointsPred;
    void Generate_sigma(void);
    // statistics tool function
    static void CalculateMeanCovariance(MatrixXf& Data, VectorXf& mean, MatrixXf& Covariance, VectorXf Weights);
};
