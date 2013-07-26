#include "emulator.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include "Eigen/Dense"
#include <ctime>	// for clock
#include <cstdlib> // for rand() and srand()
// #include <random> 	//uncomment if using getRandnum()

using namespace Eigen;
using namespace std;

//#define NDEBUG  

int main()
{
	char sim_out[255], sim_in[255], x_var[255];
	int discard, num_PC, iter, simtype;
	emulator data;
	GPdat GPfit;
	GPdat GPpred;

	cout << endl << "Read sim data from file or generate simulation? (Read = 0, generate = 1)" << endl;
	cin >> simtype;
	
	if (simtype == 0)
	{
		cout << "\nType file name for simulation output.\nFile format: \n";
		cout << "First two numbers specify row, column size. Then the matrix is output of each simulation x number of simulations.\n";
		cin >> sim_out;
		cout << "\nType file name for simulation design (# of simulations x # of paramaters).\n";
		cout << "Again, first two numbers specify row and column size.\n";
		cin >> sim_in;
		cout << "\nType file name for independent variable array. \nFirst number specifies length of array.\n";
		cin >> x_var;
	
		//	Read above files into ysim, design, and xvar
		data.readin(sim_out, sim_in, x_var);
	
	
	
		cout << "Number of simulation runs to ignore: \n";
		cin >> discard;

		data.setup(discard);
    }
    
    if (simtype == 1)
    {
    	int simchoice;
    	cout << "Please type the number corresponding to the simulation you'd like to use." << endl;
    	cout << "\t1. cos(x) + theta^2" << endl << "\t2. ????" << endl << "\t3. ????" << endl << endl;
    	cin >> simchoice;
    	data.gensim(simchoice);
    }
    

    cout << "Number of principal components: \n";
    cin >> num_PC;

	iter = 500;
	
	data.GPsetup(num_PC);
	
	GPfit = data.fitGP(iter);
	
	// cout << "lam_eta fit is: \n" << GPfit.lam_eta << endl;
	// cout << "lambdas fit is: \n" << GPfit.lambdas << endl;
	// cout << "rhos fit is: \n" << GPfit.rhos << endl;
	
	/***** make predictions *****/
	int burnin = 100;
	int interval = 20;
	int i;
	
	int numreal = ((iter - burnin) / interval) + 1;
	// cout << "numreal is: " << numreal << endl;
	GPpred.lam_eta.resize(numreal);
	GPpred.lambdas.resize(numreal, GPfit.lambdas.cols());
	GPpred.rhos.resize(numreal, GPfit.rhos.cols());
	
	for (i = 0; i < numreal ; i++)
	{
		GPpred.lam_eta(i) = GPfit.lam_eta((burnin - 1) + interval * i);
		GPpred.lambdas.row(i) = GPfit.lambdas.row((burnin - 1) + interval * i);
		GPpred.rhos.row(i) = GPfit.rhos.row((burnin - 1) + interval * i);
	}
	// cout << "lam_eta pred is: \n" << GPpred.lam_eta << endl;
	// cout << "lambdas pred is: \n" << GPpred.lambdas << endl;
	// cout << "rhos pred is: \n" << GPpred.rhos << endl;
	int predictagain = 1;
	while(predictagain)
	{
		predictagain = 1;
		data.setupdesignpred(numreal);
		for (i = 0; i < numreal; i++)
		{
			data.predGP(GPpred, i);
		}
		data.savepred();
		
		cout << "\nWould you like to make another prediction using a different set of parameters? (No = 0; Yes = 1)" << endl;
		cin >> predictagain;		
	}
	
	
	return 1;
}
