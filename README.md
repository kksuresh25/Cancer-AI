# Cancer-AI

## Introduction 
Cancer-AI is a computational platform developed for in silico profiling of activating mutations in kinases. Our platform is able to robustly predict the activating behavior of uncharacterized mutations found in the tyrosine kinase domain (TKD) of kinases implicated in cancer with a **balanced accuracy of 82%**. Cancer-AI was developed in the Radhakrishnan Lab (http://www.seas.upenn.edu/~biophys) at the University of Pennsylvania. 

## Set-up  
Before getting started, please check the `requirements.txt` to ensure that the correct version of Python packages are installed on your local machine. 

There are four folders in the directory: 

1. SVM 
2. Logistic Regression
3. Neural Network
4. Random Forest 

Each folder contains the following core code for the implementation of each ML algorithm. 

1. all_kinase.xlsx 
2. paper_train_data
3. paper_test_data 
4. constructdata.py
5. tunehyperparameters.py
6. testmodel.py 

The `all_kinase.xlsx` excel file contains information for the kinases that this algorithm can be used on. 

The `paper_train_data` text file contains the data used to train the ML algorithm. The `paper_test_data` text file contains the data used to test the ML algorithm. The example test dataset focuses on mutations found in the TKD of the Anaplastic Lymphoma Kinase (ALK). Both of these files have the following data regarding the mutation: 

1. the name of the kinase (BRAF, ALK, etc.) 
2. the wild type residue
3. the location of the point mutaiton 
4. the mutant residue 
5. label (+1: activating, -1: non-activating) 

The `constructdata.py` script contains a function for generating feature vectors for each of the mutations in the input file (`paper_train_data` and/or `paper_test_data`) and organizing this information into a data matrix that our ML algorithms can process. 

The `tunehyperparameters.py` script contains a workflow for optimizing the hyperparameters for each ML algorthm. 

The `testmodel.py` script will train the ML algorithm using the data from `paper_train_data` adn then apply this model to predcit the activating behavior of mutants provided in `paper_test_data`. The script will output a file `results.csv` that organizes the prediction results. If this script is being used for the purposes of validating the algorithm against known test data, then the following performance metrics will be displayed as well: 

1. Accuracy 
2. Balanced Accuracy 
3. Precision 
4. Recall 
5. ROC AUC 

## How to use

The following is the recommended workflow for using the Cancer-AI platform: 

1. Use the `pip install` command to ensure Python packages match those found in `requirements.txt` 
2. Download folder for ML algorithm to your local machine (SVM for example)
3. Input mutations you need predictions for in `paper_test_data` in the exact same format as the examples already entered. Make sure that the mutations are in the TKD of the kinase and that the kinase can be found in `all_kinase.xlsx`. If the mutations are uncharacterized, still enter an arbritrary label - this will not affect the predictions. 
4. Run `python2.7 svm_testmodel.py` from the terminal
5. Open output `ai_results.csv` to see predictions made by algorithm for each mutant system

## More information 
The following paper further details the methodology and applications of our platform: "Computational algorithms for in silico profiling of activating mutations in cancer" (https://doi.org/10.1007/s00018-019-03097-2) 


