#include "UKF.h"

UKF::UKF(uint8_t n)
{
    // initialisation of the instance
    this->state = VectorXf{ n };
    this->cov = MatrixXf{ n, n };

    this->n = n;
    this->n_aug = n + 2;
    this->m = 2 * this->n_aug + 1;

    // Weights of sigma points
    this->lambda = 3 - static_cast<float>(this->n_aug);

    this->w = VectorXf{ this->m };
    float t = lambda + static_cast<float>(this->n_aug);
    this->w(0) = lambda / t;
    this->w.tail(this->m - 1).fill(0.5 / t);

    this->SigmaPoints = MatrixXf{ this->n_aug, this->m };
    this->SigmaPointsPred = MatrixXf{ this->n, this->m };

    // initialisation of the state
    this->state.fill(0);
    
    this->Ax = 1000;
    this->Ay = 1000;
    this->Px = 20 * 20;         // measurement noise for x axis
    this->Py = 20 * 20;         // measurement noise for y axis

    this->cov <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    this->Q = MatrixXf{ n, n };
    this->Q <<
        10, 0, 0, 0,
        0, 200, 0, 0,
        0, 0, 10, 0,
        0, 0, 0, 200;
}

UKF::~UKF()
{

}

VectorXf UKF::model_function(VectorXf input, float dt)
{
    // implementation of the model tracking [x,y,x_dot,y_dot,x_dot_dot,y_dot_dot]T
    VectorXf output{ this->n };

    output(0) = input(0) + input(2) * dt + 0.5 * input(4) * dt * dt;
    output(1) = input(1) + input(3) * dt + 0.5 * input(5) * dt * dt;
    output(2) = input(2) + input(4) * dt;
    output(3) = input(3) + input(5) * dt;

    //output(0) = input(0);
    //output(1) = input(1);
    //output(2) = input(2);
    //output(3) = input(3);

    return output;
}

void UKF::predict(float dt)
{
    // generate sigma points
    this->Generate_sigma();

    // iterate through
    for (uint8_t i = 0; i < this->m; i++)
    {
        this->SigmaPointsPred.col(i) = this->model_function(this->SigmaPoints.col(i), dt);
    }
    // calculate new mean& covariance
    UKF::CalculateMeanCovariance(this->SigmaPointsPred, this->state, this->cov, this->w);
}

void UKF::update(VectorXf measure)
{
    const int n_z = 2;
    // convert the predicted sigma points into measure space
    MatrixXf Zsig{ n_z, this->m };
    Zsig.row(0) = this->SigmaPointsPred.row(0);     // assign x coord
    Zsig.row(1) = this->SigmaPointsPred.row(1);     // assign y coord

    VectorXf Zmean{ n_z };
    MatrixXf S{ n_z, n_z };
    MatrixXf R = MatrixXf(n_z, n_z);
    R << Px, 0,
        0, Py;

    UKF::CalculateMeanCovariance(Zsig, Zmean, S, this->w);
        //add measurement noise covariance matrix
    S = S + R;

    // calculate kalman gain

        //create matrix for cross correlation Tc
    MatrixXf Tc = MatrixXf(this->n, n_z);
        //calculate cross correlation matrix
    Tc.fill(0.0);
    for (uint8_t i = 0; i < this->m; i++)
    {
        // residual
        VectorXf z_diff = Zsig.col(i) - Zmean;
        // state difference
        VectorXf x_diff = this->SigmaPointsPred.col(i) - this->state;
        Tc = Tc + this->w(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K
    MatrixXf K = Tc * S.inverse();
    // residual
    VectorXf z_diff = measure - Zmean;
    // update state mean and covariance matrix
    this->state = this->state + K * z_diff;
    this->cov = this->cov - K * S * K.transpose();

}

void UKF::Generate_sigma(void)
{
    VectorXf x_aug = VectorXf(this->n_aug);
    x_aug << this->state, 0, 0;

    MatrixXf cov_aug{ this->n_aug, this->n_aug };
    cov_aug.fill(0);
    cov_aug.topLeftCorner(this->n, this->n) = this->cov + this->Q;
    cov_aug(this->n_aug - 2, this->n_aug - 2) = this->Ax;
    cov_aug(this->n_aug - 1, this->n_aug - 1) = this->Ay;

    MatrixXf sqrt_P = cov_aug.llt().matrixL();

    this->SigmaPoints.col(0) = x_aug;
    for (int i = 0; i < this->n_aug; i++)
    {
        // columns 1 -> n_aug_ = x + sqrt((lambda + n_aug_) * P_) 
        this->SigmaPoints.col(i + 1) = x_aug + sqrt(this->lambda + static_cast<float>(this->n_aug)) * sqrt_P.col(i);
        // columns n_aug_+1 -> 2*n_aug_+1 = x - sqrt((lambda + n_aug_) * P_)
        this->SigmaPoints.col(i + 1 + this->n_aug) = x_aug - sqrt(this->lambda + static_cast<float>(this->n_aug)) * sqrt_P.col(i);
    }

    //MatrixXf sqrt_P = this->cov.llt().matrixL();

    //this->SigmaPoints.col(0) = this->state;
    //for (int i = 0; i < this->n; i++)
    //{
    //    // columns 1 -> n_aug_ = x + sqrt((lambda + n_aug_) * P_) 
    //    this->SigmaPoints.col(i + 1) = this->state + sqrt(this->lambda + static_cast<float>(this->n)) * sqrt_P.col(i);
    //    // columns n_aug_+1 -> 2*n_aug_+1 = x - sqrt((lambda + n_aug_) * P_)
    //    this->SigmaPoints.col(i + 1 + this->n) = this->state - sqrt(this->lambda + static_cast<float>(this->n_aug)) * sqrt_P.col(i);
    //}
}

void UKF::CalculateMeanCovariance(MatrixXf& Data, VectorXf& mean, MatrixXf &Covariance, VectorXf Weights)
{
    // calculate mean
    mean.fill(0);

    for (int i = 0; i < Data.cols(); i++)
    {
        mean += Weights(i) * Data.col(i);
    }
    // calculate covariance
    Covariance.fill(0.0);
    for (int i = 0; i < Data.cols(); i++) 
    {
        // state difference
        VectorXf x_diff = Data.col(i) - mean;
        Covariance += Weights(i) * x_diff * x_diff.transpose();
    }
}
