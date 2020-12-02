# Cancer-AI

## Introduction 
Cancer-AI is a computational platform developed for in silico profiling of activating mutations in cancer. Our platform is able to robustly predict the activating behavior of mutations found in the tyrosine kinase domain (TKD) of kinases implicated in cancer with a **balanced accuracy of 82%**. Cancer-AI was developed in the Radhakrishnan Lab (http://www.seas.upenn.edu/~biophys) at the University of Pennsylvania. 

## Set-up  
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

The `all_kinase.xlsx` excel file contains information for the kinases that this algorithm can be used on. If a kinase of interest is not available in the file, feel free to manually enter the information for the system into the excel file. 

The `paper_train_data` text file contains the data used to train the ML algorithm. The `paper_test_data` text file contains the data used to test the ML algorithm. The example test dataset focuses on mutations found in the TKD of the Anaplastic Lymphoma Kinase (ALK). Both of these files have the following data regarding the mutation: 

1. the name of the kinase (BRAF, ALK, etc.) 
2. the wild type residue
3. the location of the point mutaiton 
4. the mutant residue 
5. label (+1: activating, -1: non-activating) 

The `constructdata.py` script contains a function for generating feature vectors for each of the mutations in the input file (`paper_train_data` and/or `paper_test_data`) and organizing this information into a data matrix that our ML algorithms can process. 

The `tunehyperparameters.py` script contains a workflow for optimizing the hyperparameters for each ML algorthm. 

The `testmodel.py` script will train the ML algorithm using the data from `paper_train_data` and then apply this model to predict the activating behavior of mutants provided in `paper_test_data`. The script will output a file `ai_results.csv` that organizes the prediction results. If this script is being used for the purposes of validating the algorithm against known test data, then the following performance metrics will be displayed as well: 

1. Accuracy 
2. Balanced Accuracy 
3. Precision 
4. Recall 
5. ROC AUC 

## How to use
The following is the recommended workflow for using the Cancer-AI platform:

1. Download repository to local machine 

2. Input mutations you need predictions for in `paper_test_data` in the exact same format as the examples already entered. Make sure that the mutations are in the TKD of the kinase and that the kinase can be found in `all_kinase.xlsx`. If a kinase of interest is not available in the file, feel free to manually enter the information for the system into the excel file.If the mutations are uncharacterized, still enter an arbritrary label (-1 or 1) - this will not affect the predictions. 

3. Open the terminal and run the following commands: 

   1. `pip2 install virtualenv` 
   2.  `virtualenv --python=python2 venv`
   3. `pip install -r requirements.txt`
   4. `python predict.py`
   5. `deactivate`
   
4. Open output `ai_results.csv` to see predictions made by ML algorithms for each mutant system 

## More information & Citation
The following paper further details the methodology and applications of our platform. If you find this code useful in your research, please cite: 

Jordan, E.J., Patil, K., Suresh, K. et al. Computational algorithms for in silico profiling of activating mutations in cancer. Cell. Mol. Life Sci. 76, 2663–2679 (2019). https://doi.org/10.1007/s00018-019-03097-2
