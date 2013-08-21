#include "GPemulator.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>

int main()
{
	char sim_out[255], sim_in[255], fitfile[255], otherysim[255], otherdesign[255];
	int discard, takelog, sims, num_PC, iter;
	MatrixXd ysimMat(0,0);
	MatrixXd designMat(0,0);
	emulator data;
	GPdat GPfit;
	
	// get files for ysim and design from user
	cout << endl << "Type file name for simulation output (output of simulation x # of simulations)." << endl;
	cout << "File format: first two numbers specify row, column size." << endl;
	cin >> sim_out;
	cout << endl << "Type file name for simulation design (# of simulations x # of paramaters)." << endl;
	cout << "Again, first two numbers specify row and column size." << endl;
	cin >> sim_in;
	
	//	Read above files into ysim, design
	ysimMat = ReadInMatrix(sim_out);
	designMat = ReadInMatrix(sim_in);
	
	data.initialize(ysimMat, designMat);
	
	cout << endl << "Number of simulation runs to ignore: " << endl;
	cin >> discard;
	if (discard > 0)
	{	
		data.discardsims(discard);
	}
	
	cout << endl << "Take log of sim output? (No = 0; Yes = 1)" << endl;
	cin >> takelog;
	
	while (takelog != 0 && takelog != 1)
	{
		cout << endl << "Your selection is invalid. Please choose again." << endl;
		cin >> takelog;
	}
	
	if (takelog == 1) data.takelog();
	
	cout << endl << "How many simulations do you want to feed to the emulator? (size of training set)" << endl;
	cin >> sims;	
	
	data.setup(sims);
	
	cout << endl << "Number of principal components: " << endl;
    cin >> num_PC;
    
    data.GPsetup(num_PC);
    
   	// Set priors for GP fit
    float a_eta = 1;
	float b_eta = 0.0001;
    int aw = 10;
	int bw = 10;
	float brho = 0.1;
	int seed = 43;		// seed for random # generator
	
	iter = 1500;	// number of iterations for the MCMC chain
    
    GPfit = data.fitGP(iter, b_eta, a_eta, bw, aw, brho, seed);
    
    cout << endl << "Saving GP fit data. What would you like to name the file?" << endl;
    cin >> fitfile;
    data.savefit(GPfit, fitfile);
    
    int save;
    cout << endl << "Would you like to save the validation set (sims not used to build the GP fit)? (No = 0; Yes = 1)" << endl;
    cin >> save;
    
    while (save != 0 && save != 1)
	{
		cout << endl << "Your selection is invalid. Please choose again." << endl;
		cin >> save;
	}
    
    if (save == 1)
    {
    	cout << endl << "Type file name for validation set simulator output." << endl;
    	cin >> otherysim;
    	
    	cout << endl << "Type file name for validation set input parameters." << endl;
    	cin >> otherdesign;
    	
    	data.SaveValidationData(otherysim, otherdesign);
    }
    
    cout << endl << "GP fit complete. To make prediction, run GPpredict.cpp using file: " << fitfile << endl << endl;
    
    return 1;
}

