# Cancer-AI

## Introduction 
Cancer-AI is a computational platform developed for in silico profiling of activating mutations of kinases. Our platform is able to robostly predict the activating behavior of uncharacterized mutations found in the tyrosine kinase domain (TKD) of kinases implicated in cancer. Cancer-AI was developed in the Radhakrishnan Lab (https://fling.seas.upenn.edu/~biophys/dynamic/wordpress/) at the University of Pennsylvania. 

## Set-up  
Before getting started, please check the `requirements.txt` to ensure that the correct version of Python packages are installed on your local machine. 

There are four folders in the directory: 

1. SVM 
2. Logistic Regression
3. Neural Network
4. Random Forest 

Each folder contains the following core code for the implementation of each ML algorithm. 

1. testmodel.py 
2. paper_train_data 
3. paper_test_data 
4. constructdata.py 

`paper_train_data` contains the data used to train the ML algorithm. `paper_test_data` contains the data used to test the ML algorithm. The example test dataset focuses on mutations found in the TKD of the Anaplastic Lymphoma Kinase (ALK). The `constructdata.py` script contains 

## How to use


## More information 
The following paper further details the methodology and applications of our platform: "Computational algorithms for in silico profiling of activating mutations in cancer" (https://doi.org/10.1007/s00018-019-03097-2) 


