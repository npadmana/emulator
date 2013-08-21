#ifndef GPEMULATOR_H
#define GPEMULATOR_H

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct GPdat 	// Contains information about the Gaussian process 
{
    VectorXd lam_eta;	// overall precision parameter
    MatrixXd lambdas;	// precision parameters
    MatrixXd rhos;		// correlation parameters
};

class emulator
{
	private:
		MatrixXd ysim, design, ysimValidation, designValidation, designpred, ypred, Ksim, K, invKtrK;
		VectorXd eta, what, ysimmean;
		float ysimsd;
		int nsims;
		int numPC;
		int noutput;
		int ntheta;
		MatrixXd std_design();	// Scale the design matrix
		MatrixXd std_ysim();		// Standardize the simulations
		MatrixXd makeRmat(MatrixXd, RowVectorXd);		// builds R matrix
		MatrixXd makesigmaw(MatrixXd, RowVectorXd, RowVectorXd);	// function to build sigmaw covariance matrix
		
	public:
		emulator(); 	// constructor
		void initialize(MatrixXd, MatrixXd); 	// takes ysim and design matrices as arguments. Sets ysim, design, ntheta, and noutput
		void discardsims(int);		// discard the first ____ simulation runs
		void takelog();			// take log of simulator output (ysim)
		void setup(int);	// Randomly separates sims into ysim and otherysim.
		void GPsetup(int);		// sets up K, invKtrK matrices and eta for fitGP and predGP
		GPdat fitGP(int, float, float, int, int, float, int); 	// fits the GP to training set.
		void savefit(GPdat, char*);		// save GP fit data that predGP function needs. info must be read using ReadFitData function
		void SaveValidationData(char* , char*);		// save sims and design that were not used to build the GP fit
		GPdat ReadFitData(char*);		// initialize variables using info contained in file produced by savefit function
		void setupdesignpred(int, VectorXd);	// set up designpred		
		void predGP(GPdat, int); 	// make prediction. this function calculates the ypred matrix
		VectorXd averagepreds();	// averages the different predictions made for a single set of input parameters
		MatrixXd getypred();		// returns the predictions before averaging them.
};

MatrixXd ReadInMatrix(char*);   // read in matrix from file
MatrixXd SortByCol(MatrixXd);	// uses 'sortvec' of random #s which is the last row of matrix to randomly sort columns of the matrix.
MatrixXd SortByRow(MatrixXd);	// uses 'sortvec' of random #s which is the rightmost col of matrix to randomly sort rows of the matrix
float unifrnd(float, float);	// uniform random # generator
MatrixXd invert(MatrixXd);		// invert matrix using SVD

#endif