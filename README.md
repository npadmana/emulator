emulator
========

                           Gaussian Process Emulator in C++

*********************************** Basic Overview ***********************************

Given the output of a certain number of simulations (ysim) and the input settings
at which each simulation was run (design), the emulator predicts what the simulator
might output at input settings that have not yet been tried by fitting a Gaussian 
process to the simulation output.

The GPemulator.h header file contains the functions necessary to run such a 
Gaussian process emulator.

In addition to the header file are two example programs: GPfit and GPpredict.
 
GPfit: 		fits a Gaussian process to the simulation runs included in the training set, 
			then saves all necessary information about the fit to file.
		
GPpredict:	reads information from the file saved in GPfit and makes predictions at 
			input parameter settings chosen by the user.


******************************* Parameters Set by User *******************************

- Number of principal components
- Priors for Gaussian process fit
- Random number generator seed
- Number of iterations for MCMC loop
- Burnin and interval for MCMC


******************************* Tips and Known Issues ********************************

- Matrix format: the emulator assumes that the design matrix and ysim matrix are of 
  this format:
	
          design: rows = simulation runs (nsims), columns = input parameters (ntheta)
          ysim:   rows = simulator output (noutput), columns = # of simulations (nsims)

- K matrix blows up as nsims increases. Avoid training sets of more than a few
  hundred simulation runs.

- Accuracy of predictions depends on which simulations were chosen for the training set,
  and not necessarily how many. Quality over quantity.

- When making predictions, keep MCMC interval low or = 0 if possible for more accurate 
  predictions. 

- Compile using -O3 optimizations and compiler clang++, c++, or g++ (I've found g++ 
  runs slightly slower)
