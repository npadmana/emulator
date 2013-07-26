#ifndef EMULATOR_H
#define EMULATOR_H

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include "Eigen/Dense"
#include <ctime>	// for clock
#include <cstdlib> // for rand() and srand()

using namespace Eigen;
using namespace std;

struct GPdat 
{
    VectorXd lam_eta;
    MatrixXd lambdas;
    MatrixXd rhos;
};

class emulator
{
	private:
		MatrixXd ysim, design, otherysim, otherdesign, designpred, ypred;
		MatrixXd Ksim, K, invKtrK;
		VectorXd xvar;
		VectorXd eta;
		VectorXd what;
		VectorXd ysimmean;
		float ysimsd;
		int nsims;
		int numPC;
		int noutput;
		int ntheta;
		int truepoint;
		MatrixXd std_design();	// Scale the design matrix
		MatrixXd std_ysim();		// Standardize the simulations
		MatrixXd makeRmat(MatrixXd, RowVectorXd);		// builds R matrix
		MatrixXd makesigmaw(MatrixXd, RowVectorXd, RowVectorXd);	// function to build sigmaw covariance matrix
		
	public:
		emulator(); 	// constructor
		void gensim(int);	// Simple function to generate fake simulator response. 
   		void readin(char*, char*, char*);    // read in ysim, design, and xvar from three files
		void setup(int);	// Randomly separates sims into ysim and otherysim. Takes 'discard' as an argument = the # of sims to ignore.
		void GPsetup(int);		// Sets up K, invKtrK matrices and eta for fitGP and predGP
		GPdat fitGP(int); 	// fits the GP to training set
		void setupdesignpred(int);		// Choose prediction parameters and set up designpred		
		void predGP(GPdat, int); // make prediction. this function calculates the ypred matrix
		void savepred();	// save ypred to file formatted for plotting
};

MatrixXd SortByCol(MatrixXd);	// Uses 'sortvec' of random #s which is the last row of matrix to sort columns of the matrix.
MatrixXd SortByRow(MatrixXd);	// Uses 'sortvec' of random #s which is the rightmost col of matrix to sort rows of the matrix
float unifrnd(float, float);	// Uniform random # generator
MatrixXd invert(MatrixXd);		// Invert matrix using SVD

#endif