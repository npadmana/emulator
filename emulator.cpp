#include "emulator.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include "Eigen/Dense"
#include <ctime>	// for clock
#include <cstdlib> // for rand() and srand()

using namespace Eigen;
using namespace std;

emulator::emulator()
{
	ysim.resize(0,0);
	design.resize(0,0);
	otherysim.resize(0,0);
	otherdesign.resize(0,0);
	Ksim.resize(0,0);
	K.resize(0,0);
	invKtrK.resize(0,0);
	ypred.resize(0,0);
	designpred.resize(0,0);
	xvar.resize(0);
	eta.resize(0);
	what.resize(0);	
	ysimmean.resize(0);
	ysimsd = 0.0;
	nsims = 0;
	numPC = 0;
	noutput = 0;
	ntheta = 0;
	truepoint = 0;
}

// Simple function to generate fake simulator response. 
void emulator::gensim(int test)
{
	// one parameter design: cos(x) + theta^2
	if(test == 1)
	{
		int i, j;
		
		ntheta = 1;
		float thetamin = 0.0;
		float thetamax = 1.0;
		float dtheta = 0.1;
		nsims = (thetamax - thetamin)/dtheta + 1;
		design.resize(nsims,ntheta);
		design(0,0) = thetamin;
		for (i = 1; i < nsims; i++)
		{
			design(i,0) = design(i-1,0) + dtheta;
		}
		
		float xmin = 0.0;
		float xmax = 2.0;
		float dx = 0.1;
		noutput = (xmax - xmin) / dx + 1;
		xvar.resize(noutput);
		xvar(0) = xmin;
		for (i = 1; i < noutput; i++)
		{
			xvar(i) = xvar(i-1) + dx;
		}
		
		ysim.resize(noutput, nsims);
		
		for (int i = 0; i < noutput; i++)
		{
			for(int j = 0; j < nsims; j++)
			{
				ysim(i,j) = cos(xvar(i)) + pow(design(j,0),2);
			}
		}
	}
	
	cout << "design is: \n" << design << endl;
	cout << "xvar is: \n" << xvar << endl;
	cout << "ysim is: \n" << ysim << endl;
 
}	

void emulator::readin(char* filename1, char* filename2, char* filename3)
{
	int sims;
	
	// first read in x-var data
	ifstream x_var;
    x_var.open(filename3);
    
    if(x_var.fail()) 
    {
		cout << "Can't open file: " << filename3 << endl;
		exit(1);
    }
    
    // first number in file is size of sim output
    x_var >> noutput;
    xvar.resize(noutput);
    //	Read in from file
	int i;
	float x;
	for(i = 0; i < noutput; i++)
	{
		x_var >> x;
		xvar(i) = x;
	}
	x_var.close();

	cout << "\nX variable array size is: " << xvar.size() << endl;
	cout << "X variable array is:\n" << xvar << endl;
	
	// now read in simulation output
	ifstream sim_out;
	sim_out.open(filename1);
	
	if(sim_out.fail())
	{
		cout << "Can't open file: " << filename1 << endl;
		exit(1);	
	}
	
	// number of rows = size of sim output, number of columns = number of simulation runs
	sim_out >> noutput >> sims;
	
	ysim.resize(noutput, sims);
	int j;
	for (i = 0; i < noutput; i++)
	{
		for (j = 0; j < sims; j++)
		{
			sim_out >> x;
			ysim(i,j) = x;
		}	
	}
	sim_out.close();
	
	cout << "\nysim matrix (format: output of simulations x number of simulations)\n" << ysim << endl;
	
	// now read in simulation design
	ifstream sim_in;
	sim_in.open(filename2);
	
	if(sim_in.fail())
	{
		cout << "Can't open file: " << filename2 << endl;
		exit(1);
	}
	
	// number of rows = number of simulation runs, number of cols = number of parameters
	sim_in >> sims >> ntheta;
	design.resize(sims, ntheta);
	for (i = 0; i < sims; i++)
	{
		for (j = 0; j < ntheta; j++)
		{
			sim_in >> x;
			design(i,j) = x;
		}
	}
	sim_in.close();
	
	cout << "\nDesign matrix is:\n" << design << endl;
	cout << "\nTotal number of simulations is: " << ysim.cols() << endl;
	cout << "Number of parameters is: " << design.cols() << endl;
}

//	Sets ysim, design (the training set), and otherysim and otherdesign
void emulator::setup(int discard)
{
	// 	Ignore the first "discard" simulations specified by user
	ysim = ysim.rightCols(ysim.cols() - discard);
	design = design.bottomRows(design.rows() - discard);
	
	// cout << "\nysim matrix minus discarded columns is: \n" << ysim << endl;
	// cout << "\nDesign matrix minus discarded rows is:\n" << design << endl;
	cout << endl << ysim.cols() << " simulations remain." << endl;

	int ans = 0;
	cout << endl << "Multiply sim output by r^2? (No = 0; Yes = 1)" << endl;
	cin >> ans;
	
	if (ans == 1)
	{
		int j;
		for (j = 0; j < ysim.cols(); j++)
		{
			ysim.col(j) = ysim.col(j).array() * xvar.array().pow(2);
		}
		cout << "check!" << endl;
	}
	
	ans = 0;
	cout << endl << "Take log of x and sim output? (No = 0; Yes = 1)" << endl;
	cin >> ans;
	if (ans == 1)
	{
		ysim = ysim.array().log();
		xvar = xvar.array().log();
		cout << "check!";
	}

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
	
	//	Separate data into training set and validation set
	cout << "\nHow many simulations do you want to feed to the emulator? (size of training set)\n";
	cin >> nsims;	
	
	otherysim = ysim.rightCols(ysim.cols() - nsims);
	ysim = ysim.leftCols(nsims);

	otherdesign = design.bottomRows(design.rows() - nsims);
	design = design.topRows(nsims);	
	
	// cout << "\nYsim is: \n" << ysim << endl;
	// cout << "\notherysim is \n" << otherysim << endl;
	// cout << "\ndesign is: \n" << design << endl;
	// cout << "\notherdesign is: \n" << otherdesign << endl;	
}

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

/* 
Matlab code to standardize ysim
ysimmean = mean(ysim,2);
ysim0 = ysim - repmat(ysimmean,[1 nsims]);
ysimsd = sqrt(var(ysim0(:)));
ysimStd = ysim0./ysimsd;
display(ysimStd) 
*/
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
	MatrixXd design_std(nsims, ntheta);
	MatrixXd ysimStd(noutput, nsims);

	numPC = num_PC;
	//	Scale the design matrix and standardize ysim
	design_std = std_design();
	ysimStd = std_ysim();
	
	// cout << "\nThis is ysim standardized: " << endl << ysimStd << endl;
	// cout << "\nThis is design standardized: " << endl << design_std << endl;
	Ksim.resize(nsims,numPC);
	
	//	Find principal component representation
	JacobiSVD<MatrixXd> svd(ysimStd, ComputeThinU | ComputeThinV);
	Ksim.noalias() = (1/sqrt(nsims)) * svd.matrixU().leftCols(numPC) * (svd.singularValues().head(numPC).asDiagonal());
	// cout << "\nThis is Ksim: " << endl << Ksim << endl;
	
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
	// cout << "\nK matrix is: " << endl << K << endl;	
		
	//	Define eta (stack up sims--stack columns of matrix into a vector)
	eta.resize(noutput * nsims);
	for (i = 0; i < nsims; i++)
	{
		eta.segment(i * noutput, noutput) = ysimStd.col(i);
	}
	
	// cout << "eta vector is: " << endl << eta << endl;
	
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

GPdat emulator::fitGP(int niter)
{
	GPdat fitGPout;
	//default_random_engine rd(2094);		// uncomment if using getRandnum
	MatrixXd design_std(nsims, ntheta);
	design_std = std_design();
	
	int neta_tot = noutput * nsims;
	
	float a_eta = 1;
	float b_eta = 0.001;
	
	float a_eta_p = a_eta + nsims * (float)(noutput - numPC) / 2;
	
	MatrixXd I(neta_tot, neta_tot);

	VectorXd tmpvec(eta.size());
	tmpvec.noalias() = K.transpose() * eta;
	
	float b_eta_p = b_eta + 0.5 * (eta.squaredNorm() - tmpvec.dot(invKtrK * tmpvec));
	// cout << "b_eta_p is: \n" << b_eta_p << endl;
	
	int aw = 10;
	int bw = 10;
	float brho = 0.1;
	
	
	/******* SET UP MCMC *************************/
	RowVectorXd mux(1 + numPC + numPC * ntheta);
	mux(0) = 10000;				// lambda_eta, the overall precision parameter
	mux.segment(1,numPC).setOnes();		// lamw, initialize precision params
	mux.segment(numPC+1, numPC * ntheta).setConstant(0.5);	// rhow, initialize correlation params
	
	// cout << "mux is: \n" << mux << endl;
	
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
	// cout << "This is r: \n" << r << endl;
	
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
	srand(43);
	float logrand;
	for(count = 0; count < niter; count++)
	{
		time_t t1 = clock(); // start timer
		for(component = 0; component < nk; component++)
		{
			dmux.setZero();
			dux = unifrnd(-r(component), r(component));		// generate a proposal
			// cout << "dux is: " << dux << endl;
			dmux(component) = dux;
			trial.noalias() = mux + dmux;
			proposals++;

			/***** computation of log likelihood *******/
			reject = 1; 	// flag to reject trial, default to reject until checks made
			// cout << "Trial is: \n" << trial << endl;
			if (trial(0) >= 0 && (trial.segment(1, numPC).array() > 0).all() && (trial.segment(numPC + 1, numPC * ntheta).array() > 0).all() && (trial.segment(numPC + 1, numPC * ntheta).array() < 1).all())
			{
// 				if((trial.segment(1, numPC).array() > 0).all())		// lamw check
// 				{
// 					if((trial.segment(numPC + 1, numPC * ntheta).array() > 0).all())
// 					{
// 						if((trial.segment(numPC + 1, numPC * ntheta).array() < 1).all())	// rhow check
// 						{
// 							reject = 0;
// 						}
// 					}
// 				}
				reject = 0;
			}
			// cout << "reject is: " << reject << endl;
		
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
				// cout << "log ratio is: " << logratio << endl;
				logrand = log(unifrnd(0.0,1.0)+eps);
				if (logratio >= logrand)
				{
					accept++;
					mux = trial;
				}
			}		
		}
		
		sample.row(count) = mux;
		time_t t2 = clock();
		cout << "Time elapsed: " << (double)(t2 - t1) / (double) CLOCKS_PER_SEC << endl;
	}
	
	// cout << "proposals is: " << proposals << endl;
	//	recall nk = (1 + numPC + numPC * ntheta) = sample.rows() = mux.size()
	fitGPout.lam_eta.resize(niter);
	fitGPout.lambdas.resize(niter, numPC);
	fitGPout.rhos.resize(niter, numPC * ntheta);
	
	/***** end loop control *****/
	// float acceptance_rate = (float) accept / proposals;
	// cout << "Acceptance rate is: " << acceptance_rate << endl;
		
	fitGPout.lam_eta = sample.col(0);
	fitGPout.lambdas.noalias() = sample.block(0,1,niter,numPC);
	fitGPout.rhos.noalias() = sample.rightCols(numPC*ntheta);
	
	return fitGPout;
}

void emulator::setupdesignpred(int numreal)
{
	ypred.resize(noutput, numreal);
	ypred.setZero();
	int i;	
	RowVectorXd thetapred(ntheta);	
	
	if(otherdesign.rows() == 0 && otherdesign.cols() == 0)
	{
		cout << "\nPlease type the parameter settings you'd like to use, pressing 'SPACE' after each number.\n";
		for (i = 0; i < ntheta; i++)
		{
			cin >> thetapred(i);
		}
		cout << "Emulator will make prediction for input settings: " << thetapred << endl;
		truepoint--;
	}
	
	else
	{
		MatrixXd display(otherdesign.rows(), otherdesign.cols() + 1);

		for (i = 0; i < otherdesign.rows(); i++)
		{
			display(i,0) = i + 1;
		}
		display.block(0,1, otherdesign.rows(), otherdesign.cols()) = otherdesign;
		cout << "\nSimulation design for validation data (runs not included in training set): \n" << display << endl;
		cout << "Please select from the above the parameter setting you'd like "; 
		cout << "to use for the prediction by typing the corresponding number in the leftmost column. ";
		cout << "OR type 0 if you would like to use a new set of parameters not included in the list above.\n";
		cin >> truepoint;
	
		while (truepoint < 0 || truepoint > otherdesign.rows())
		{
			cout << "The choice you selected is invalid. Please select again." << endl;
			cin >> truepoint;
		}
	
		if (truepoint > 0)
		{
			truepoint--;
			thetapred = otherdesign.row(truepoint);
	
			cout << endl << "You chose the setting: " << thetapred << endl;
			cout << "\nThis parameter setting corresponds to the actual simulation output:\n" << otherysim.col(truepoint) << endl;
		}
	
		if(truepoint == 0)
		{
			cout << "\nPlease type the parameter settings you'd like to use, pressing 'SPACE' after each number.\n";
			for (i = 0; i < ntheta; i++)
			{
				cin >> thetapred(i);
			}
			cout << "Emulator will make prediction for input settings: " << thetapred << endl;
			truepoint--;
		}
	}	
	
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

void emulator::predGP(GPdat GPpred, int k)
{
	RowVectorXd lamws(GPpred.lambdas.cols());
	RowVectorXd rhows(GPpred.rhos.cols());
	
	// float lameta = GPpred.lambdas(k);	
	lamws.noalias() = GPpred.lambdas.row(k);
	rhows.noalias() = GPpred.rhos.row(k);
	
	/******** now build the covariance matrix and take submatrices **********/
	MatrixXd sigmapred(designpred.rows() * numPC, designpred.rows() * numPC);
	sigmapred.setZero();
	sigmapred = makesigmaw(designpred, lamws, rhows); 	// This has all the principal components sigmas on the diagonal
	
	// cout << "Sigmapred is: \n" << sigmapred << endl;
	
	RowVectorXd wpred(numPC);
	int nsims_pred = designpred.rows();
	int p;
	MatrixXd sigtmp(sigmapred.rows(), sigmapred.cols());
	MatrixXd sig11(nsims, nsims);
	MatrixXd sig21(nsims_pred - nsims, nsims);
	// float temp;
	MatrixXd tempmat(1,1);
	for (p = 0; p < numPC; p++)
	{
		sigtmp = sigmapred.block(p * nsims_pred, p * nsims_pred, nsims_pred, nsims_pred); 
		sig11 = sigtmp.block(0,0,nsims,nsims);
    	sig21 = sigtmp.block(nsims,0, sigtmp.rows() - nsims, nsims);
		tempmat.noalias() = sig21 * invert(sig11) * what.segment(p * nsims, nsims);	
		wpred(p) = tempmat(0,0);
	}
	
	// cout << "wpred is: \n" << wpred << endl;
	
	/************** Make prediction *************/
	
	// First on std scale
	for (p = 0; p < numPC; p++)
	{
		ypred.col(k) += wpred(p) * Ksim.col(p);
	}
	
	// Now on native scale
	ypred.col(k) = ypred.col(k) * ysimsd + ysimmean;
	 
}

void emulator::savepred()
{
	cout << endl << "ypred is: \n" << ypred << endl;
	VectorXd ypredmean(noutput);
	ypredmean = ypred.rowwise().mean();
	cout << endl << "ypred mean is: \n" << ypredmean << endl;
		
	ofstream myfile;
	char filename[255];
	
	cout << "\nSaving ypred. What would you like to name the file?" << endl;
	cin >> filename;
	
	int i, j;
	
	//	Calculate the mean sim output and its std deviation
	VectorXd ysimstddev(noutput);
	MatrixXd diff_from_mean(noutput, nsims);

	for (i = 0; i < ysim.rows(); i++)
	{
		for (j = 0; j < ysim.cols(); j++)
		{
			diff_from_mean(i, j) = pow(ysim(i, j) - ysimmean(i),2);
		}
	}
	
	ysimstddev = diff_from_mean.rowwise().mean();
	for (i = 0; i < ysimstddev.size(); i++)
	{
		ysimstddev(i) = sqrt(ysimstddev(i));
	}
	
	myfile.open (filename);	
	
	// Store all data in file
	myfile << "#xvar\t" << "#ysim mean\t" << "#stddev\t" << "#pred output\n" << "#actual output\t";		// column headings
	for (i = 0; i < ypred.rows(); i++)
	{
		j = 0;
		myfile << xvar(i) << "\t";
		myfile << ysimmean(i) << "\t" << ysimstddev(i) << "\t" << ypred(i,j);
		if (ypred.cols() > 1)
		{
			for (j = 1; j < ypred.cols(); j++)
			{
				myfile << "\t" << ypred(i,j);
			}
		}
		if(otherdesign.rows() != 0)
		{
			myfile << "\t" << otherysim(i,truepoint);
		}
		myfile << endl;
	}
	myfile.close();
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

// float getRandnum(default_random_engine rd, float min, float max)
// {
//   // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
// 	//default_random_engine rd(43);
// 	uniform_real_distribution<float> distribution(min,max);
// 
// 	return distribution(rd);
// }