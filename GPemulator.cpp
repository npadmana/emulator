#include "GPemulator.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <ctime>	// for clock
#include <iomanip> // for setprecision (when saving file)

emulator::emulator()
{
	ysim.resize(0,0);
	design.resize(0,0);
	ysimValidation.resize(0,0);
	designValidation.resize(0,0);
	Ksim.resize(0,0);
	K.resize(0,0);
	invKtrK.resize(0,0);
	ypred.resize(0,0);
	designpred.resize(0,0);
	eta.resize(0);
	what.resize(0);	
	ysimmean.resize(0);
	ysimsd = 0.0;
	nsims = 0;
	numPC = 0;
	noutput = 0;
	ntheta = 0;
}

//	Sets ysim, design, ntheta, noutput, and nsims
void emulator::initialize(MatrixXd ysimMat, MatrixXd designMat)
{
	ysim = ysimMat;
	design = designMat;
	
	ntheta = design.cols();
	noutput = ysim.rows();
	nsims = ysim.cols();
}

// 	Discard the first "ignore" simulation runs specified by the user
void emulator::discardsims(int ignore)
{
	ysim = ysim.rightCols(ysim.cols() - ignore);
	design = design.bottomRows(design.rows() - ignore);
	
	nsims = ysim.cols();
	
	cout << endl << nsims << " simulations remain." << endl;
}

//	Takes log of simulator output (ysim)
void emulator::takelog()
{
	ysim = ysim.array().log();
}

//	Sets up ysim, design (the training set), and ysimValidation and designValidation
void emulator::setup(int sims)
{		
	int total_sim = ysim.cols();
	ifstream randnum;
	VectorXd sortvec;
	
	//	Read in a list of random numbers as the vector to sort the simulations
	sortvec.resize(total_sim);
	randnum.open("randomnumbers1.txt",ios::in);
	int i, x;
	for (i = 0; i < total_sim; i++)
	{
		randnum >> x;
		sortvec(i) = x;	
	}
	randnum.close();
	
	//	Append sortvec to the last row of ysim and the last column of design
	ysim.conservativeResize(ysim.rows()+1, NoChange);
	design.conservativeResize(NoChange, design.cols()+1);
	
	ysim.row(ysim.rows()-1) = sortvec.transpose();
	design.col(design.cols()-1) = sortvec;
	
	//	Sort the rows or columns of the matrices according to sortvec
	ysim = SortByCol(ysim);
	design = SortByRow(design);
		
	//	Delete sortvec from matrices
	ysim.conservativeResize(ysim.rows()-1, NoChange);
	design.conservativeResize(NoChange, design.cols()-1);

	//	reset nsims
	nsims = sims;
	
	//	Separate data into training set and validation set	
	ysimValidation = ysim.rightCols(ysim.cols() - nsims);
	ysim = ysim.leftCols(nsims);

	designValidation = design.bottomRows(design.rows() - nsims);
	design = design.topRows(nsims);	
}

// standardize the design matrix
MatrixXd emulator::std_design()
{
	MatrixXd newmat(nsims,ntheta);
	RowVectorXd colmax(ntheta);
	RowVectorXd colmin(ntheta);
	int i,j;
	
	//	Find min and max values in each column and put into two row-vectors
	colmin = design.colwise().minCoeff();
	colmax = design.colwise().maxCoeff();	
	
	//	Now scale the matrix
	for (i = 0; i < nsims; i++)
	{
		for (j = 0; j < ntheta; j++)
		{
			newmat(i,j) = (design(i,j) - colmin(j)) / (colmax(j) - colmin(j));
		}
	}
	
	return newmat;
}

// standardize sims
MatrixXd emulator::std_ysim()
{
	MatrixXd ysim0(noutput, nsims);
	MatrixXd ysimStd(noutput, nsims);

	
	ysimmean.resize(noutput);

	int i,j;
	float sum = 0;

	ysimmean = ysim.rowwise().mean();

	for(j = 0; j < nsims; j++)
	{
		ysim0.col(j) = ysim.col(j) - ysimmean;
	}

	//	Calculate the variance over entire ysim0 matrix.
	float mean = ysim0.mean();
	for (i = 0; i < noutput; i++)
	{
		for (j = 0; j < nsims; j++)
		{
			sum = pow(ysim0(i,j) - mean, 2) + sum; 
		}
	}
	ysimsd = sqrt(sum / float (noutput * nsims));	

	ysimStd = ysim0 / ysimsd;
	
	return ysimStd;
}

// function to set up K matrix
void emulator::GPsetup(int num_PC)
{
	numPC = num_PC;	
	
	MatrixXd design_std(nsims, ntheta);
	MatrixXd ysimStd(noutput, nsims);
	
	//	Scale the design matrix and standardize ysim
	design_std = std_design();
	ysimStd = std_ysim();
	
	Ksim.resize(nsims,numPC);
	
	//	Find principal component representation
	JacobiSVD<MatrixXd> svd(ysimStd, ComputeThinU | ComputeThinV);
	Ksim.noalias() = (1/sqrt(nsims)) * svd.matrixU().leftCols(numPC) * (svd.singularValues().head(numPC).asDiagonal());
	
	//	Create overall K matrix with tensor (Kronecker) products
	K.resize(nsims*Ksim.rows(), nsims*numPC);
	int i,j;
	VectorXd ksim(nsims*Ksim.rows());
	for(i = 0; i < numPC; i++)
	{
		for(j = 0; j < nsims; j++)
		{
			ksim.setZero(nsims * Ksim.rows());
			ksim.segment(Ksim.rows()*j,Ksim.rows()) = Ksim.col(i);
			K.col(nsims*i + j) = ksim;
		}
	}	
		
	//	Define eta (stack up sims--stack columns of matrix into a vector)
	eta.resize(noutput * nsims);
	for (i = 0; i < nsims; i++)
	{
		eta.segment(i * noutput, noutput) = ysimStd.col(i);
	}
	
	//	Find inverse (K' * K) = invKtrK
	invKtrK.noalias() = K.transpose() * K;
	invKtrK = invKtrK.inverse();
	
	//	Define 'what' vector
	what.resize(invKtrK.rows());
	what.noalias() = invKtrK * K.transpose() * eta;	
}

// function to build R matrix
// takes standardized design and correlation params on the PC weights
// and generates an nsim by nsim matrix that feeds into the overall covariance matrix
MatrixXd emulator::makeRmat(MatrixXd design_std, RowVectorXd rhos)
{
	int n_sims = design_std.rows();
	MatrixXd Rmat(n_sims, n_sims);
	MatrixXd tmp(n_sims, n_sims);
	Rmat.setZero();
	Rmat.triangularView<Upper>().setOnes();		// since Rmat is symmetric, only fill in upper triangular part
	
	int k, l, p;
	float factor;
	for (k = 0; k < n_sims - 1; k++)
	{
		for (l = k + 1; l < n_sims; l++)
		{
			factor = 1.0;
			for (p = 0; p < ntheta; p++)
			{
				factor = factor * pow(rhos(p),4*pow(design_std(k,p) - design_std(l,p),2)); // correlation function
			}
			Rmat(k,l) = factor;
		}
	}
	
	// take everything above diagonal (not including diagonal), transpose
	tmp = Rmat.triangularView<StrictlyUpper>().transpose(); 
	Rmat += tmp;	// fill in lower triangular part

	return Rmat;
}

// function to build sigmaw covariance matrix
// takes standardized design and precision/correlation params on the PC weights...
// ...and generates an nsim*numPC by nsim*numPC matrix
MatrixXd emulator::makesigmaw(MatrixXd design_std, RowVectorXd lamw, RowVectorXd rhow)
{
	int n_sims = design_std.rows();
	MatrixXd sigmaw(n_sims * numPC, n_sims * numPC);
	MatrixXd tmp(n_sims, n_sims);
	RowVectorXd rhop(ntheta);
	
	sigmaw.setZero();
	int p;
	for (p = 0; p < numPC; p++)
	{
		rhop = rhow.segment(p * ntheta, ntheta);				// take all the rhos corresponding to p'th PC
		tmp.noalias() = (1/lamw(p)) * makeRmat(design_std, rhop);			// apply covariance rule
		sigmaw.block(p * n_sims, p * n_sims, n_sims, n_sims) = tmp;	// fill up diagonals of sigmaw
	}

	MatrixXd I(n_sims * numPC, n_sims * numPC);
	/***** Add ridge to sigmaw *****/
	sigmaw += I.setIdentity() * 0.00000001;
	
	return sigmaw;
}

GPdat emulator::fitGP(int niter, float b_eta = 0.0001, float a_eta = 1, int bw = 10, int aw = 10, float brho = 0.1, int seed = 43)
{
	GPdat fitGPout;
	MatrixXd design_std(nsims, ntheta);
	design_std = std_design();
	
	int neta_tot = noutput * nsims;
	
	float a_eta_p = a_eta + nsims * (float)(noutput - numPC) / 2;
	
	MatrixXd I(neta_tot, neta_tot);

	VectorXd tmpvec(eta.size());
	tmpvec.noalias() = K.transpose() * eta;
	
	float b_eta_p = b_eta + 0.5 * (eta.squaredNorm() - tmpvec.dot(invKtrK * tmpvec));

	/******* SET UP MCMC *************************/
	cout << endl << "Starting MCMC for fitGP..." << endl << endl;	
	
	RowVectorXd mux(1 + numPC + numPC * ntheta);
	mux(0) = 10000;				// lambda_eta, the overall precision parameter
	mux.segment(1,numPC).setOnes();		// lamw, initialize precision params
	mux.segment(numPC+1, numPC * ntheta).setConstant(0.5);	// rhow, initialize correlation params
	
	int nk = mux.size();
	
	//	Trial vector
	RowVectorXd trial(nk);
	trial = mux;
	
	int accept = 0;			// proposals accepted
	int proposals = 0;		// proposal counter
	
	RowVectorXd r(nk);		// r = proposal width in standardized space
	r(0) = 100;
	r.segment(1,numPC).setConstant(0.1);
	r.segment(numPC + 1, numPC * ntheta).setConstant(0.1);
	
	MatrixXd sample(niter, nk);		// dimension sample array
	sample.setZero();
	float eps = 0.0000000001;		// small number
	
	/**** loop control *****/
	int count, component, reject, k, l;
	float dux, lam_eta_new, lam_eta_old, lamwfacnew, lamwfacold, rhonew, rhoold, rhofacnew, rhofacold, logratio;
	RowVectorXd dmux(nk);
	RowVectorXd lamwnew(numPC);
	RowVectorXd lamwold(numPC);
	RowVectorXd rhownew(numPC * ntheta);
	RowVectorXd rhowold(numPC * ntheta);
	MatrixXd sigtotnew(nsims * numPC, nsims * numPC);
	MatrixXd sigtotold(nsims * numPC, nsims * numPC);
	srand(seed);
	float logrand;
	time_t t1, t2, T1, T2;
	T1 = clock(); 	// start timer for entire MCMC
	for(count = 0; count < niter; count++)
	{
		t1 = clock(); // start timer for one iteration of MCMC
		for(component = 0; component < nk; component++)
		{
			dmux.setZero();
			dux = unifrnd(-r(component), r(component));		// generate a proposal
			dmux(component) = dux;
			trial.noalias() = mux + dmux;
			proposals++;

			/***** computation of log likelihood *******/
			reject = 1; 	// flag to reject trial, default to reject until checks made
			if (trial(0) >= 0 && (trial.segment(1, numPC).array() > 0).all() && (trial.segment(numPC + 1, numPC * ntheta).array() > 0).all() && (trial.segment(numPC + 1, numPC * ntheta).array() < 1).all())
			{
				reject = 0;
			}
		
			if(reject == 0)
			{	
				lam_eta_new = trial(0);
				lamwnew.noalias() = trial.segment(1, numPC);
				rhownew.noalias() = trial.segment(numPC + 1, numPC * ntheta);
				sigtotnew.noalias() = (1/lam_eta_new) * invKtrK + makesigmaw(design_std,lamwnew, rhownew);
				lam_eta_old = mux(0);
				lamwold.noalias() = mux.segment(1, numPC);
				rhowold.noalias() = mux.segment(numPC + 1, numPC * ntheta);
				lamwfacnew = 0;
				lamwfacold = 0;
				rhofacnew = 0;
				rhofacold = 0;
				for (k = 0; k < numPC; k++)
				{
					lamwfacnew += (aw - 1) * log(lamwnew(k)) - bw * lamwnew(k);
					lamwfacold += (aw - 1) * log(lamwold(k)) - bw * lamwold(k);
					for (l = 0; l < ntheta; l++)
					{
						rhonew = trial(numPC + 1 + k * ntheta + l);
						rhofacnew += (brho - 1) * log(1 - rhonew);
						rhoold = mux(numPC + 1 + k * ntheta + l);
						rhofacold += (brho - 1) * log(1 - rhoold);
					}
				}	
				sigtotold.noalias() = (1/lam_eta_old) * invKtrK + makesigmaw(design_std, lamwold, rhowold);
				logratio = 	-0.5 * log(sigtotnew.determinant() + eps) \
							+0.5 * log(sigtotold.determinant() + eps) \
							-0.5 * what.transpose() * (sigtotnew.colPivHouseholderQr().solve(what)) \
							+0.5 * what.transpose() * (sigtotold.colPivHouseholderQr().solve(what)) \
							+(a_eta_p - 1) * log(lam_eta_new) \
							-(a_eta_p - 1) * log(lam_eta_old) \
							-b_eta_p * lam_eta_new \
							+b_eta_p * lam_eta_old \
							+lamwfacnew \
							-lamwfacold \
							+rhofacnew \
							-rhofacold;
				logrand = log(unifrnd(0.0,1.0)+eps);
				if (logratio >= logrand)
				{
					accept++;
					mux = trial;
				}
			}		
		}
		
		// recall nk = (1 + numPC + numPC * ntheta) = sample.rows() = mux.size()		
		sample.row(count) = mux;
		t2 = clock(); 	// end timer
		cout << "Iteration " << count + 1 << ": " << (double)(t2 - t1) / (double) CLOCKS_PER_SEC << " seconds elapsed." << endl;
	}
	
	T2 = clock();	// end timer
	
	fitGPout.lam_eta.resize(niter);
	fitGPout.lambdas.resize(niter, numPC);
	fitGPout.rhos.resize(niter, numPC * ntheta);
	
	/***** end loop control *****/
	float acceptance_rate = (float) accept / proposals;
	cout << endl << "Acceptance rate is: " << acceptance_rate << endl;
	cout << endl << "Total MCMC took " << (double)(T2 - T1) / (double) CLOCKS_PER_SEC << " seconds to complete." << endl;
		
	fitGPout.lam_eta = sample.col(0);
	fitGPout.lambdas.noalias() = sample.block(0,1,niter,numPC);
	fitGPout.rhos.noalias() = sample.rightCols(numPC*ntheta);
	
	return fitGPout;
}

// saves all fit information needed to make prediction to file. Read in this file using ReadFitData function
void emulator::savefit(GPdat GPfit, char* filename)
{
	ofstream myfile;
	int i,j;

	myfile.open (filename);	
	
	// Store all data in file
	myfile << "#This file will only be used by function emulator::ReadFItData. ";
	myfile << "It contains all information necessary to run the prediction separately from the fit program." << endl;
	
	// Save lam_eta
	myfile << "#lam_eta" << endl;
	myfile << GPfit.lam_eta.size() << endl;;
	for (i = 0; i < GPfit.lam_eta.size(); i++)
	{
		myfile << scientific << setprecision(30) << GPfit.lam_eta(i) << endl;
	}
	
	// Save lambdas
	myfile << "#lambdas" << endl;
	myfile << GPfit.lambdas.rows() << " " << GPfit.lambdas.cols() << endl;
	for (i = 0; i < GPfit.lambdas.rows(); i++)
	{
		for (j = 0; j < GPfit.lambdas.cols(); j++)
		{
			myfile << GPfit.lambdas(i,j) << " ";
		}
		myfile << endl;	
	}
	
	// Save rhos
	myfile << "#rhos" << endl;
	myfile << GPfit.rhos.rows() << " " << GPfit.rhos.cols() << endl;
	for (i = 0; i < GPfit.rhos.rows(); i++)
	{
		for (j = 0; j < GPfit.rhos.cols(); j++)
		{
			myfile << GPfit.rhos(i,j) << " ";
		}
		myfile << endl;	
	}
	
	// Save ysim -- simulations in training set.
	myfile << "#ysim" << endl;
	myfile << ysim.rows() << " " << ysim.cols() << endl;
	for (i = 0; i < ysim.rows(); i++)
	{
		for (j = 0; j < ysim.cols(); j++)
		{
			myfile << ysim(i,j) << " ";
		}
		myfile << endl;
	}
	
	// Save design -- simulation design for the training set
	myfile << "#design" << endl;
	myfile << design.rows() << " " << design.cols() << endl;
	for (i = 0; i < design.rows(); i++)
	{
		for (j = 0; j < design.cols(); j++)
		{
			myfile << design(i,j) << " ";
		}
		myfile << endl;
	}
	
	myfile.close();
}

// If a validation set exists, save it
void emulator::SaveValidationData(char* otherysim, char* otherdesign)
{
	if (designValidation.rows() == 0 || ysimValidation.cols() == 0)
	{
		cout << endl << "Sorry, the validation set is empty so there is no information to save to file." << endl;
	}
	
	else
	{
		int i,j;
		ofstream ysimfile;
		ysimfile.open(otherysim);
		// Save ysimValidation -- simulations in the validation set
		ysimfile << "#Validation set simulator output" << endl;
		ysimfile << ysimValidation.rows() << " " << ysimValidation.cols() << endl;
		for (i = 0; i < ysimValidation.rows(); i++)
		{
			for (j = 0; j < ysimValidation.cols(); j++)
			{
				ysimfile << scientific << setprecision(30) << ysimValidation(i,j) << " ";
			}
			ysimfile << endl;
		}
		ysimfile.close();
	
		ofstream designfile;
		designfile.open(otherdesign);
	
		// Save designValidation -- simulation design for the validation set
		designfile << "#Validation set input parameter design" << endl;
		designfile << designValidation.rows() << " " << designValidation.cols() << endl;
		for (i = 0; i < designValidation.rows(); i++)
		{
			for (j = 0; j < designValidation.cols(); j++)
			{
				designfile << scientific << setprecision(30) << designValidation(i,j) << " ";
			}
			designfile << endl;
		}
		designfile.close();
	}
}

// Reads in data contained in file produced by savefit function
GPdat emulator::ReadFitData(char* filename)
{
	GPdat GPpred;
	int nrows, ncols;
	
	ifstream fitfile;
	fitfile.open(filename);
	
	if(fitfile.fail())
	{
		cout << endl << "Can't open file: " << filename << endl;
		exit(1);	
	}
	
    string line;
    int i, j;
    
    getline(fitfile, line);		// don't read commented line
    getline(fitfile, line);		// don't read commented line
    
    // Read in lam_eta
    fitfile >> nrows;
 	GPpred.lam_eta.resize(nrows);
    for (i = 0; i < nrows; i++)
    {
    	fitfile >> GPpred.lam_eta(i);	
    }
	getline(fitfile, line);    // Move to next line
    
    // Read in lambdas
	getline(fitfile, line);    // don't read commented line   
	fitfile >> nrows >> ncols;
	GPpred.lambdas.resize(nrows, ncols);
	for (i = 0; i < nrows; i++)
	{
		for (j = 0; j < ncols; j++)
		{
			fitfile >> GPpred.lambdas(i,j);
		}
	}
	getline(fitfile, line);    // move to next line	
    
    // Read in rhos
	getline(fitfile, line);    // don't read commented line       
	fitfile >> nrows >> ncols;
	GPpred.rhos.resize(nrows, ncols);
	for (i = 0; i < nrows; i++)
	{
		for (j = 0; j < ncols; j++)
		{
			fitfile >> GPpred.rhos(i,j);
		}
	}
	getline(fitfile, line);    // move to next line 
    
    // Read in ysim
	getline(fitfile, line);    // don't read commented line       
	fitfile >> nrows >> ncols;
	ysim.resize(nrows, ncols);
	for (i = 0; i < nrows; i++)
	{
		for (j = 0; j < ncols; j++)
		{
			fitfile >> ysim(i,j);
		}
	}
	getline(fitfile, line);    // move to next line		
    
    // Read in design
	getline(fitfile, line);    // don't read commented line   
	fitfile >> nrows >> ncols;
	design.resize(nrows, ncols);
	for (i = 0; i < nrows; i++)
	{
		for (j = 0; j < ncols; j++)
		{
			fitfile >> design(i,j);
		}
	}
    
    fitfile.close();
    
	ntheta = design.cols();
	nsims = design.rows();
	noutput = ysim.rows();
	numPC = GPpred.lambdas.cols();
	
	return GPpred;
}

// Initializes the prediction design
void emulator::setupdesignpred(int numreal, VectorXd thetapred)
{
	ypred.resize(noutput, numreal);
	ypred.setZero();
	int i;
	
	/**** standardize the prediction points ****/
	RowVectorXd thetapred_std(ntheta);
	thetapred_std = thetapred;
	
	RowVectorXd colmin(ntheta);
	RowVectorXd colmax(ntheta);
	
	colmin = design.colwise().minCoeff();
	colmax = design.colwise().maxCoeff();
	
	for (i = 0; i < ntheta; i++)
	{
		thetapred_std(i) = (thetapred(i) - colmin(i)) / (colmax(i) - colmin(i)); 
	}	
	
	// ... and add to the design matrix
	designpred.resize(nsims + 1, ntheta);
	designpred << std_design(), thetapred_std;
}

// make prediction using kth set of fit parameters. predicted output becomes the kth column of ypred
void emulator::predGP(GPdat GPpred, int k = 1)
{
	RowVectorXd lamws(GPpred.lambdas.cols());
	RowVectorXd rhows(GPpred.rhos.cols());
	
	lamws = GPpred.lambdas.row(k);
	rhows = GPpred.rhos.row(k);
	
	/******** now build the covariance matrix and take submatrices **********/
	MatrixXd sigmapred(designpred.rows() * numPC, designpred.rows() * numPC);
	sigmapred.setZero();
	sigmapred = makesigmaw(designpred, lamws, rhows); 	// This has all the principal components sigmas on the diagonal
	
	RowVectorXd wpred(numPC);
	int nsims_pred = designpred.rows();
	int p;
	MatrixXd sigtmp(sigmapred.rows(), sigmapred.cols());
	MatrixXd sig11(nsims, nsims);
	MatrixXd sig21(nsims_pred - nsims, nsims);
	MatrixXd tempmat(1,1);
	for (p = 0; p < numPC; p++)
	{
		sigtmp = sigmapred.block(p * nsims_pred, p * nsims_pred, nsims_pred, nsims_pred); 
		sig11 = sigtmp.block(0,0,nsims,nsims);
    	sig21 = sigtmp.block(nsims,0, sigtmp.rows() - nsims, nsims);
		tempmat.noalias() = sig21 * invert(sig11) * what.segment(p * nsims, nsims);	
		wpred(p) = tempmat(0,0);
	}
	
	/************** Make prediction *************/
	
	// First on std scale
	for (p = 0; p < numPC; p++)
	{
		ypred.col(k) += wpred(p) * Ksim.col(p);
	}
	
	// Now on native scale
	ypred.col(k) = ypred.col(k) * ysimsd + ysimmean;
}

// average each row of ypred to get the mean predicted output
VectorXd emulator::averagepreds()
{
	VectorXd meanpred(ypred.rows());
	
	meanpred = ypred.rowwise().mean();
	
	return meanpred;
}

// returns ypred -- the matrix of predictions made using different sets of fit parameters.
MatrixXd emulator::getypred()
{
	return ypred;
}

// Reads data from file into a matrix. First number of file = # of rows, second number of file = # of columns
MatrixXd ReadInMatrix(char* filename)
{
	ifstream matrixfile;
	matrixfile.open(filename);
	
	if(matrixfile.fail())
	{
		cout << endl << "Can't open file: " << filename << endl;
		exit(1);	
	}
	
	int nrows, ncols;
	
	// read in the number of rows and number of columns
	matrixfile >> nrows >> ncols;
	
	//	read into matrix M
	float x;
	MatrixXd M(nrows,ncols);
	int i, j;
	for (i = 0; i < nrows; i++)
	{
		for (j = 0; j < ncols; j++)
		{
			matrixfile >> x;
			M(i,j) = x;
		}	
	}
	matrixfile.close();

	return M;
}


//	Sort columns of matrix based on values in bottom row
MatrixXd SortByCol(MatrixXd num)
{
     int i, flag = 1, numLength = num.cols();
     int d = numLength;
     VectorXd temp(num.rows());
     while(flag || (d > 1))      // boolean flag (true when not equal to 0)
     {
          flag = 0;           // reset flag to 0 to check for future swaps
          d = (d+1) / 2;
          for (i = 0; i < (numLength - d); i++)
        {
               if (num(num.rows()-1,i + d) < num(num.rows()-1,i))
              {
                      temp = num.col(i + d);      // swap positions i+d and i
                      num.col(i + d) = num.col(i);
                      num.col(i) = temp;
                      flag = 1;                  // tells swap has occurred
              }
         }
     }
     return num;
}

//	Sort rows of matrix based on values in rightmost column
MatrixXd SortByRow(MatrixXd num)
{
     int i,flag = 1, numLength = num.rows();
     int d = numLength;
     RowVectorXd temp(num.cols());
     while( flag || (d > 1))      // boolean flag (true when not equal to 0)
     {
          flag = 0;           // reset flag to 0 to check for future swaps
          d = (d+1) / 2;
          for (i = 0; i < (numLength - d); i++)
        {
               if (num(i + d, num.cols()-1) < num(i, num.cols()-1))
              {
                      temp = num.row(i + d);      // swap positions i+d and i
                      num.row(i + d) = num.row(i);
                      num.row(i) = temp;
                      flag = 1;                  // tells swap has occurred
              }
         }
     }
     return num;
}

// Generate a random number between min and max (inclusive)
float unifrnd(float min, float max) 
{
    return ((double)rand() / ((double)RAND_MAX)) * (max - min) + min;
}

// Compute matrix inverse with SVD
MatrixXd invert(MatrixXd A)
{
	JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
	
	int numdiag = svd.singularValues().size();
	VectorXd D(numdiag);
	MatrixXd L(svd.matrixU().rows(), svd.matrixU().cols());
	MatrixXd R(svd.matrixV().rows(), svd.matrixV().cols());
	
	D.noalias() = svd.singularValues();
	L.noalias() = svd.matrixU();
	R .noalias() = svd.matrixV();
	
	RowVectorXd keep(numdiag);
	
	int i;
	int k = 0;
	for (i = 0; i < numdiag; i++)
	{
		if (D(i) > 0.000001)
		{
			keep(k) = i;
			k++;
		}
	}
	
		
	MatrixXd newD(k, k);
	MatrixXd newL(L.rows(), k);
	MatrixXd newR(R.rows(), k);
	
	newD.setZero();
	for (i = 0; i < k; i++)
	{
		newD(i,i) = D(keep(i));
	}
	
	newL = L.leftCols(keep(k - 1)+1);
	newR = R.leftCols(keep(k - 1)+1);
	
	MatrixXd invA(R.rows(), L.rows());
	invA.noalias() = newR * newD.inverse() * newL.transpose();
	
	return invA;
}
