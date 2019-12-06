# import python libraries
import os
import numpy as np
import pandas as pd
import pickle as pk
import xgboost as xgb
from sklearn.model_selection import train_test_split

# import modules from utils
from utils import equal_class 

class DataLoader():
    '''A class for loading and transforming data for the nil-pick models '''

    def __init__(self, dataframe):
        self.df=dataframe           

    def preprocess_numerical(self, cols):
        proc_numer_data = []
        for col in cols:       
            max_ = np.max(self.df[col]) 
            min_ = np.min(self.df[col]) 
            print(min_, max_)
            normalised_col = [ (float(p-min_) / float(max_-min_) ) for p in self.df[col] ] 
            proc_numer_data.append(normalised_col)
            if not os.path.exists('./info_from_preproc/sanity_checks/'+str(col) ): 
               os.makedirs('./info_from_preproc/sanity_checks/'+str(col) )  
            np.savetxt('./info_from_preproc/sanity_checks/'+str(col) +'/raw_numer_'+ str(col), self.df[col], newline='\n', fmt='%s' )
            np.savetxt('./info_from_preproc/sanity_checks/'+str(col) +'/proc_numer_'+str(col), normalised_col, newline='\n', fmt='%s' )         
        proc_numer_data=np.array(proc_numer_data).T 
        np.savetxt('./info_from_preproc/proc_numer_data', proc_numer_data, newline='\n', fmt='%s' )          
        print(proc_numer_data.shape)
        return np.array(proc_numer_data)

    def preprocess_categorical(self, cols): 
        proc_cat_data=[]      
        for col in cols:
            uniques=self.df[col].unique() 
            print(uniques) 
            np.savetxt('./info_from_preproc/uniques/'+str(col), uniques, newline='\n', fmt='%s' )
            a=pd.get_dummies(self.df[col])
            b=np.fliplr(a.values)  
            if not os.path.exists('./info_from_preproc/sanity_checks/'+str(col) ): 
               os.makedirs('./info_from_preproc/sanity_checks/'+str(col) )
            np.savetxt('./info_from_preproc/sanity_checks/'+str(col)+'/b_raw_cat_'+ str(col), b, newline='\n', fmt='%s' )
            L=[]
            print('processing col: ', col)  
            for i in range(self.df.shape[0]):
                l=list(b[i])               
                l_=l.index(1)
                L.append(l_/float(len(l)-1))  
            proc_cat_data.append(L)
            np.savetxt('./info_from_preproc/sanity_checks/'+str(col)+ '/raw_cat_'+ str(col), self.df[col], newline='\n', fmt='%s' )
            np.savetxt('./info_from_preproc/sanity_checks/'+str(col)+ '/proc_cat_'+str(col), L, newline='\n', fmt='%s' )         
        proc_cat_data=np.array(proc_cat_data).T
        np.savetxt('./info_from_preproc/proc_cat_data', proc_cat_data, newline='\n', fmt='%s' )  
        print(proc_cat_data.shape)
        return np.array(proc_cat_data)

    def preprocess_target(self, cols): 
        proc_target_data=[]
        uniques=self.df[cols[0]].unique()                                                         # [PICK, NILPICK, CNCL, ACTUALPICK, SUBS, OVRRIDE]
        print(uniques) 
        if not os.path.exists('./info_from_preproc/uniques/'+str(cols[0]) ): 
           os.makedirs('./info_from_preproc/uniques/'+str(cols[0]) )
        np.savetxt('./info_from_preproc/uniques/'+str(cols[0])+'/'+str(cols[0]), uniques, newline='\n', fmt='%s' )
        a=pd.get_dummies(self.df[cols[0]])
        b=np.fliplr(a.values)  
        np.savetxt('./info_from_preproc/uniques/'+str(cols[0])+'/b_'+str(cols[0]), b, newline='\n', fmt='%s' )
        L=[]
        print('processing col: ', cols[0])  
        for i in range(self.df.shape[0]):
            l=list(b[i])               
            l_=l.index(1)
            if l_==0 or l_==2:                      # l_=0 is 'SUBS' and l_=0.4 is 'NILPICKS'
               L.append(1)
            elif l_==1 or l_==3 or l_==4:
               L.append(0)   
            else:
               L.append(-1)
        proc_target_data=np.array(L).T         
        print(proc_target_data.shape)
        if not os.path.exists('./info_from_preproc/sanity_checks/'+str(cols[0]) ): 
           os.makedirs('./info_from_preproc/sanity_checks/'+str(cols[0]) )
        np.savetxt('./info_from_preproc/sanity_checks/raw_target', self.df[cols[0]], newline='\n', fmt='%s' )
        np.savetxt('./info_from_preproc/sanity_checks/proc_target', proc_target_data, newline='\n', fmt='%s' ) 
        print(np.unique(proc_target_data))
        proc_target_data=proc_target_data.reshape((len(proc_target_data),1))
        return np.array(proc_target_data)

    def create_train_dct(self, train_array, clusters):
        dct = {}
        for c in clusters: 
            dct['uni_%s' % c] = []                                     # initialize the key for each cluster in the dictionary
        print('Creating dictionary...') 
        for i in range(self.df.shape[0]):                              # loop over all data rows of feature "mdse_cat_nbr"
            cl=self.df["mdse_catg_nbr"][i]                             # category_nbr will become the key in the dicitonary
            dct['uni_%s' % cl].append(train_array[i])                  # append i_th entry to the key, specific to the category-cluster             
        print('The dictionary has been created!') 
        for j in range(len(dct)):
            print('the length of cluster: ', clusters[j], 'is: ', len(dct['uni_%s' % clusters[j] ]) )
        with open('./info_from_preproc/cluster_dictionary.pkl', 'wb') as f:
             pk.dump(dct, f, pk.HIGHEST_PROTOCOL)
        return dct
 
    def read_dct(self, dictionary, cluster_number):                    # read data for a particular cluster 
        print('This is cluster: ', cluster_number)                     # limit the number of rows in cluster data
        X=np.array(dictionary['uni_%s'%cluster_number])                # Define the data for cluster of number "cluster number"
        new_data, new_target, LP_rat = equal_class.equal(X)            # call utils function to create 'equal_class'
        new_data=np.array(new_data)
        new_target=np.array(new_target)
        new_target=new_target.reshape((len(new_target), 1))                
        print(new_data.shape, new_target.shape)
        if new_data.shape[0]>0:     
           X=np.hstack([new_data, new_target])
           np.random.shuffle(X)                                           # Shuffle the rows of that particular lane
           print(X.shape)
           Train=X[:, :-1]                                                # Define train and target values
           Train_T=X[:, -1:]
           print(Train.shape, Train_T.shape)                              # print dimensions for check
           X_train, X_test, y_train, y_test = train_test_split(Train, Train_T, test_size=0.2, random_state=42)   #split data to train and test sets
           return X_train, X_test, y_train, y_test, LP_rat
        else:
           print('The train array had zero rows.. skipping. ..')  
           X_train, X_test, y_train, y_test, LP_rat=[],[],[],[],0
           return X_train, X_test, y_train, y_test, LP_rat




