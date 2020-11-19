#import libraries
import numpy as np
import pandas as pd

#input: data in form of text file, number of features (59)
#output: numpy table with 59 features for each mutant system
def constructdata(dataname,numfeatures):
    #read document
    with open(dataname) as f:
        lines = f.readlines()

    #initialize vectors to hold data
    wildtype = []
    mutant = []
    labels = []
    protnames = []
    resnum = []
    features = np.zeros((len(lines),numfeatures))
    allk = pd.read_excel("all_kinase.xlsx","Sheet1")

    #initialize vectors and variables that will be used to make features matrix
    aacids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    aacids_cat = ["aliphatic","polar","acidic","acidic","aromatic","unique","basic","aliphatic","basic","aliphatic","aliphatic","polar","unique","polar","basic","polar","polar","aliphatic","aromatic" ,"aromatic"]
    aacids_hyropathy = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
    hydropathy_maxdif = max(aacids_hyropathy) - min(aacids_hyropathy)
    aacids_freeenergy = [-0.368,4.530,2.060,1.770,1.060,-0.525,0.000,0.791,0.000,1.070,0.656,0.000,-2.240,0.731,-1.030,-0.524,0.000,0.401,1.600,4.910]
    freeenergy_maxdif = max(aacids_freeenergy) - min(aacids_freeenergy)
    aacids_nVDWrad = [1.00,2.43,2.78,3.78,5.89,0.00,4.66,4.00,4.77,4.00,4.43,2.95,2.72,3.95,6.13,1.60,2.60,3.00,8.08,6.47]
    nVDWrad_maxdif = max(aacids_nVDWrad) - min(aacids_nVDWrad)
    aacids_charge = [0,0,-1,-1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0]
    charge_maxdif = max(aacids_charge) - min(aacids_charge)
    aacids_polarity = [8.1,5.5,13.0,12.3,5.2,9.0,10.4,5.2,11.3,4.1,5.7,11.6,8.0,10.5,10.5,9.2,8.6,5.9,5.4,6.2]
    polarity_maxdif = max(aacids_polarity) - min(aacids_polarity)

    #extract protein names
    for i in range(0,len(lines)):
        words = lines[i].split()
        protnames.append(words[0])


    #extract residue numbers
    for i in range(0,len(lines)):
        words = lines[i].split()
        resnum.append(words[2])

    #extract label values
    for i in range(0,len(lines)):
        words = lines[i].split()
        labels.append(words[4])

    #extract wildtype and mutant amino acids
    for i in range(0,len(lines)):
        words = lines[i].split()
        wildtype.append(words[1])
        mutant.append(words[3])

    #put label, wildtype, and mutant data in vectors
    wildtype = np.array(wildtype)
    mutant = np.array(mutant)
    labels = np.array(labels)
    resnum = np.array(resnum)
    protnames = np.array(protnames)
    labels = [int(numeric_string) for numeric_string in labels]
    resnum = [int(numeric_string) for numeric_string in resnum]

    #Add features to features matrix!
    for ii in range(0,len(mutant)):
        for jj in range(0,len(aacids)):
            if wildtype[ii] == aacids[jj]:

                #Adding features 0-19
                features[ii,jj] = 1

                #Adding features 40-44
                if aacids_cat[jj] == "aliphatic":
                    features[ii,40] = 1
                if aacids_cat[jj] == "acidic":
                    features[ii,41] = 1
                if aacids_cat[jj] == "basic":
                    features[ii,42] = 1
                if aacids_cat[jj] == "aromatic":
                    features[ii,43] = 1
                if aacids_cat[jj] == "polar":
                    features[ii,44] = 1

    for ii in range(0,len(mutant)):
        for jj in range(len(aacids),2*len(aacids)):
            if mutant[ii] == aacids[jj-20]:

                #Adding features 20-39
                features[ii,jj] = 1

                #Adding features 45-49
                if aacids_cat[jj-20] == "aliphatic":
                    features[ii,45] = 1
                if aacids_cat[jj-20] == "acidic":
                    features[ii,46] = 1
                if aacids_cat[jj-20] == "basic":
                    features[ii,47] = 1
                if aacids_cat[jj-20] == "aromatic":
                    features[ii,48] = 1
                if aacids_cat[jj-20] == "polar":
                    features[ii,49] = 1

    #Adding feature 50-54
    for ii in range(0,len(mutant)):
        for jj in range(0,len(aacids)):
            if wildtype[ii] == aacids[jj]:
                hydrowild = aacids_hyropathy[jj]
                freewild = aacids_freeenergy[jj]
                nVDWwild = aacids_nVDWrad[jj]
                chargewild = aacids_charge[jj]
                polaritywild = aacids_polarity[jj]

            if mutant[ii] == aacids[jj]:
                hydromutant = aacids_hyropathy[jj]
                freemutant = aacids_freeenergy[jj]
                nVDWmutant = aacids_nVDWrad[jj]
                chargemutant = aacids_charge[jj]
                polaritymutant = aacids_polarity[jj]

        features[ii,50] = (hydrowild-hydromutant) / hydropathy_maxdif
        features[ii,51] = (freewild-freemutant) / freeenergy_maxdif
        features[ii,52] = (nVDWwild-nVDWmutant) / nVDWrad_maxdif
        features[ii,53] = (chargewild-chargemutant) / charge_maxdif
        features[ii,54] = (polaritywild-polaritymutant) / polarity_maxdif

    #Adding features 55-58
    for ii in range(0,len(protnames)):
        for jj in range(0,len(allk['Protein_name'])):
            if protnames[ii] == allk['Protein_name'][jj]:
                if (resnum[ii] >= allk['ploop_start'][jj]) & (resnum[ii] <= allk["ploop_end"][jj]):
                    features[ii,55] = 1
                if (resnum[ii] >= allk['alphac_start'][jj]) & (resnum[ii] <= allk["alphac_end"][jj]):
                    features[ii,56] = 1
                if (resnum[ii] >= allk['catloop_start'][jj]) & (resnum[ii] <= allk["catloop_end"][jj]):
                    features[ii,57] = 1
                if (resnum[ii] >= allk['activation_start'][jj]) & (resnum[ii] <= allk["activation_end"][jj]):
                    features[ii,58] = 1

    # #Adding features 59-62
    # for ii in range(0,len(protnames)):
    #     f = open(os.path.join('phd',protnames[ii] + '.prof'), "r")
    #     lines = f.readlines()
    #     for i in range(0,len(lines)):
    #         words = lines[i].split()
    #         if words[0] == "No":
    #             newinit = i
    #             for jj in range(0,len(words)):
    #                 if words[jj] == "pH":
    #                     pHnum = jj
    #                 if words[jj] == "pE":
    #                     pEnum = jj
    #                 if words[jj] == "pL":
    #                     pLnum = jj
    #                 if words[jj] == "PACC":
    #                     PACCnum = jj
    #     for i in range(newinit+1,len(lines)):
    #         words = lines[i].split()
    #         if int(words[0]) == resnum[ii]:
    #             features[ii,59] = int(words[pHnum])
    #             features[ii,60] = int(words[pEnum])
    #             features[ii,61] = int(words[pLnum])
    #             features[ii,62] = int(words[PACCnum])

    #return data matrix and labels
    return features,labels
