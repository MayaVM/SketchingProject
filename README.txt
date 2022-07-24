# Final project is a python code meant to reproduce the experiments from the paper "Randomized Dimensionality Reduction for k-Means Clustering"
 by Christos Boutsidis, Anastasios Zouzias, Michael W. Mahoney, and Petros Drineas.
 
### File content
The file contains:
1. 4 Python files containing the code needed to re-run the experiments.
2. A pdf file containing a summary of the paper and the results of running the experiments.

### Data sets

The experiments were run on the following Kaggle data-sets:
1. USPS
2. war_news
3. words-by-partsOfSpeech

and on the coil-20 and ORL data-set, found in https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php
 and https://cam-orl.co.uk/facedatabase.html respectively.

All of the processing of the data is done by the DataLoader.py in this file.
### Code

## Needed modules
To run the Python code successfully, the following modules need to be installed:
numpy, sklearn.cluster, sklearn.datasets, matplotlib.pyplot, pandas, collections, h5py, and PIL. Please make sure they are installed before attempting to run the code.

## Python files

For convenience, the code was divided into the following files:
1. Algogithms.py - Consisting of the algorithms tested in the paper.
2. DataLoader.py - Consists of the methods to clean and process the datasets.
3. Eval.py - Consisting of the methods used to test and evaluate the different algorithms.
4. Main.py  - This file loads the datasets and runs the test and produced graphs accordingly. The graph is then saved in the same repository of the project in both
	png and eps file types. After a graph is run successfully, the console will print the test according to the dataset in the console.
 
## Running the experiments
The python files are built in a way that running main will reproduce the entire experiment set. To run changes, you can alter the following:
1. Change the algorithms: You can either add or detract the algorithm from alg_set and alg_names list accordingly. You can add new algorithms provide that they produce
	result of the correct shape Numpy array.
2. Change the data: You can either add or detract the dataset from data_set, data_names, and data_val lists accordingly. You can add new datasets provide that they produce
	result of the same shape Numpy array.
3. Change the dimensions: You can change the dimension list, providing the new list is of positive integers.
3. Change the tests: You can either add or detract the test from the test list. Note that unlike the previous two, adding a new test to the list will cause
	the code to fail. As noted in the pdf, the accuracy test had some problems regarding its results. The code includes 2 variations of the test, both with issues. 

Note: There is a problem with running algorithm 3 variations on the textual data (extended upon in the pdf file). Due to that, when running the objective value test
	on the textual datasets, you need to remove the mentioned algorithm before running the test.
