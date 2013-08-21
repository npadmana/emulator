#include "GPemulator.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <iomanip>

void SavePred(char*, MatrixXd, char*);

int main()
{
	char fitfile[255], thetafile[255], savefile[255];
	emulator data;
	GPdat GPfit, GPburn, GPpred;
	MatrixXd thetaMat(0,0);
	MatrixXd predOutput(0,0);
	RowVectorXd thetapredict(1);
	VectorXd tempvec(1);
	int predorder, iter, numreal, ans, npredict, nthetas, output, i, j;
	
	cout << endl << "Please type the name of the file containing the GP fit data." << endl;
	cin >> fitfile;
	
	GPfit = data.ReadFitData(fitfile);
	cout << endl << "GP fit data has been read in successfully." << endl;
	
	iter = GPfit.lam_eta.size();	// number of MCMC iterations
	
	data.GPsetup(GPfit.lambdas.cols());
	
	// **** make predictions *****/
	int burnin = 100;	// ignore the first 'burnin' iterations of the MCMC
	int interval = 3;  	// thinning interval --- only used if (predorder = 2) make many predictions, then average.
 	

	cout << endl << "Would you like to: (please select the corresponding number)" << endl;
	cout << "1. Average the GP fit parameters, then make one prediction (fast method)" << endl;
	cout << "2. Make a prediction using each set of fit parameters, then average the predictions (slow if # of simulations or # of iterations for MCMC chain are large)" << endl;
	cin >> predorder;
	
	while (predorder != 1 && predorder != 2)
	{
		cout << endl << "Your selection is invalid. Please choose again." << endl;
		cin >> predorder;
	}

	// Fast method: average parameters then make one prediction
	if (predorder == 1)
	{
		GPburn.lam_eta.resize(iter - burnin);
		GPburn.lambdas.resize(iter - burnin, GPfit.lambdas.cols());
		GPburn.rhos.resize(iter - burnin, GPfit.rhos.cols());

		// Ignore the first 'burnin' iterations of the MCMC
		GPburn.lam_eta = GPfit.lam_eta.tail(iter - burnin);
		GPburn.lambdas = GPfit.lambdas.bottomRows(iter - burnin);
		GPburn.rhos = GPfit.rhos.bottomRows(iter - burnin);
	
		GPpred.lam_eta.resize(1);
		GPpred.lambdas.resize(1, GPfit.lambdas.cols());
		GPpred.rhos.resize(1, GPfit.rhos.cols());
		
		//	Then average the GP fit parameters
		GPpred.lam_eta(0) = GPburn.lam_eta.mean();
		GPpred.lambdas = GPburn.lambdas.colwise().mean();
		GPpred.rhos = GPburn.rhos.colwise().mean();
	
		numreal = GPpred.lam_eta.size();
	}

	if (predorder == 2)
	{
		numreal = ((iter - burnin) / interval) + 1;
		cout << endl << numreal << " predictions will be made and then averaged for each set of input parameters" << endl;

		GPpred.lam_eta.resize(numreal);
		GPpred.lambdas.resize(numreal, GPfit.lambdas.cols());
		GPpred.rhos.resize(numreal, GPfit.rhos.cols());

		// Subtract burnin then use the parameter values for every 'interval'th MCMC iteration to make a prediction.
		for (i = 0; i < numreal ; i++)
		{
			GPpred.lam_eta(i) = GPfit.lam_eta(burnin + interval * i - 1);
			GPpred.lambdas.row(i) = GPfit.lambdas.row(burnin + interval * i -1);
			GPpred.rhos.row(i) = GPfit.rhos.row(burnin + interval * i -1);
		}
	}
	
	cout << endl << "What input parameters would you like to use for the prediction?" << endl;
	cout << "Type 1 to input a set of parameters from the keyboard." << endl;
	cout << "Or type 2 to read in several sets of parameters from a file." << endl;
	cin >> ans;
	
	while (ans != 1 && ans != 2)
	{
		cout << endl << "Your selection is invalid. Please choose again." << endl;
		cin >> ans;
	}
	
	//	Set up thetapredict (row vector containing input parameters to be used for prediction)
	nthetas = GPfit.rhos.cols() / GPfit.lambdas.cols();
	thetapredict.resize(nthetas);
	
	/**** make prediction *****/
	int predictagain = 1;
	while(predictagain)
	{
		predictagain = 0;
		
		// Get thetapredict from keyboard input
		if (ans == 1)
		{
			cout << endl << "Please type the value for each of the " << nthetas << " parameters at which you'd like to make a prediction." << endl;
			for (i = 0; i < nthetas; i++)
			{
				cin >> thetapredict(i);
			}
			cout << endl << "Emulator will make prediction for input parameter setting: " << thetapredict << endl;
		
			data.setupdesignpred(numreal, thetapredict);
			for (i = 0; i < numreal; i++)
			{
				data.predGP(GPpred, i);
			}
			tempvec = data.averagepreds();
			output = tempvec.size();
			predOutput.resize(output, npredict);
			predOutput.col(0) = data.averagepreds();		
		}

		// Read thetapredict from file
		if (ans == 2)
		{
			cout << endl << "Please type the name of the file containing the input parameters for prediction." << endl;
			cout << "File format: first number = # of rows, second number = # of cols." << endl;
			cout << "Matrix rows = number of cases to predict, columns = number of parameters" << endl;
			cin >> thetafile;
		
			thetaMat = ReadInMatrix(thetafile);
			npredict = thetaMat.rows(); cout << endl << npredict << " sets of parameters" << endl;
			for (i = 0; i < npredict; i++)
			{
				thetapredict = thetaMat.row(i);
				data.setupdesignpred(numreal, thetapredict);
				for (j = 0; j < numreal; j++)
				{
					data.predGP(GPpred, j);
				}
				// find size of simulator output
				if (i == 0)
				{
					tempvec = data.averagepreds();
					output = tempvec.size();
					predOutput.resize(output, npredict);
				}
				predOutput.col(i) = data.averagepreds();
			}
		}
		
		cout << endl << "Predicted output is: " << endl << predOutput << endl;
	
		// Save prediction to file	
		cout << endl << "Saving emulator's prediction for simulator output. What would you like to name the file?" << endl;
		cin >> savefile;
		SavePred(savefile, predOutput, fitfile);
		
		if (ans == 1) // if using keyboard input for thetapredict, give the option to make another prediction.
		{
			cout << endl << "Would you like to make another prediction using a different set of input parameters? (No = 0; Yes = 1)" << endl;
			cin >> predictagain;
			while (predictagain != 0 && predictagain != 1)
			{
				cout << endl << "Your selection is invalid. Please type 0 for 'No' or 1 for 'Yes'." << endl;
				cin >> predictagain;
			}
		}
	}
	
	cout << endl << "Goodbye!" << endl << endl;
	
	return 1;	
}

// Save predictions to file
void SavePred(char* filename, MatrixXd prediction, char* fitdatafile)
{
	ofstream myfile;

	int i, j;
	int nrows = prediction.rows();
	int ncols = prediction.cols();
	
	myfile.open (filename);	
	
	// Store all data in file
	myfile << "#Emulator prediction made from GP fit data file: " << fitdatafile << endl;
	myfile << "#nrows = " << nrows << " = size of simulator output, ncols = " << ncols << " = number of predictions made" << endl;
	for (i = 0; i < nrows; i++)
	{
		for (j = 0; j < ncols; j++)
		{
			myfile << scientific << setprecision(30) << prediction(i,j) << "\t";
		}
		myfile << endl;
	}
	myfile.close();
}