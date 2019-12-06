__author__ = 'Yuri Lebedev'
__version__ = '1.0.0'

import os
import os.path
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from core.data_processor import DataLoader
from core.xgb_model import XGB_Model
#from core.cluster_model import Cluster_Model

# import modules from utils
from utils import equal_class 

def main(do_preprocessing, do_clustering, do_dictionary, do_xgboost):

    #########################################################
    ### Load configs, Read csv, Create dataframe ############
    #########################################################
    configs = json.load(open('config.json', 'r'))                                                           # load json configuration file, that points to data csv file
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])          # 
    print('Reading csv file: '+ str(configs['data']['filename']) + ' reading...')
    dataframe = pd.read_csv( os.path.join('data', configs['data']['filename']) )                            # load csv file
    print('Done!')
    data=DataLoader(dataframe)                                                                              # define the dataframe as 'data'
    
    #########################################################
    ### Conditions on preprocessing #########################
    #########################################################
    if do_preprocessing is True:                                                                              
        numerical_data   = data.preprocess_numerical(configs['data']['columns']['numerical'] )              # preprocess numerical data
        categorical_data = data.preprocess_categorical(configs['data']['columns']['categorical'])           # preprocess categorical data
        target           = data.preprocess_target(configs['data']['columns']['target'] )                    # preprocess target data
        #category_nbr     = data.preprocess_categorical(configs['data']['columns']['category_nbr'] )         # categorize 'category_nbr'
        #department_nbr   = data.preprocess_categorical(configs['data']['columns']['department_nbr'] )       # categorize 'department_nbr'
        #department_desc  = data.preprocess_categorical(configs['data']['columns']['department_desc'] )      # categorize 'department_desc'
        #store_id         = data.preprocess_categorical(configs['data']['columns']['store_id'] )             # categorize 'store_id'      
        #time_stamps_data = data.preprocess_time_stamps_data(configs['data']['columns']['time_stamps'] )
        #multiplicity     = data.preprocess_multiplicity(configs['data']['columns']['multiplicity'] )    
        
        print(numerical_data.shape, categorical_data.shape, target.shape)
        train_array = np.hstack([numerical_data, categorical_data, target])                                 # stack the numer, categ, target pre-processed values 
        print(train_array.shape)
        np.savetxt('./info_from_preproc/train_array.txt', train_array, newline='\n', fmt='%s')              # create and save training array 
    else:
        print('preprocessing was not enabled in this run')

    #########################################################
    ### Conditions on clustering ############################
    #########################################################   

    clusters=dataframe['mdse_catg_nbr'].unique()

    if do_clustering is True:
        print('Keeping the clusters...') 
        #print('Running the clustering algorithm...')
        #cluster_model = Cluster_Model()                                                              # Load the "Clustering_Model" class
        #clusters = cluster_model.run_clustering(       )                                             # run_clustering algorithm and output clusters
        print('Done!')    
    else:
        new_clusters=[]
        for i in range(len(clusters)): 
           new_clusters=new_clusters + clusters[i]
        clusters=new_clusters   

    print(clusters) 

    #########################################################
    ### Conditions on dictionary ############################
    #########################################################    
    if do_dictionary is True:  
        print('Creating a dictionary based on the clusters formed by the clustering algorithm...')
        if do_preprocessing is False:
           print('Assuming you ran preprocessing before, you are now loading the existing train_array.txt file...')  
           train_array=np.loadtxt('./info_from_preproc/train_array.txt') 
           print('Done!')   
        dictionary = data.create_train_dct(train_array, clusters)                                           
    else:  
        print('Assuming a dictionary already exists, you are now loading the existing cluster_dictionary.pkl dictionary...')
        dictionary = joblib.load(open('./info_from_preproc/cluster_dictionary.pkl', 'rb'))                   
        print('Done!')

    #########################################################
    ### Conditions on xgboost ###############################
    #########################################################
    if do_xgboost is True:
        xgb_model = XGB_Model()                                                                       # Load the 'XGB_Model' class 
        for cl in clusters:    
            if len(dictionary['uni_%s'%cl ]) >= 1000:                                        # loop over all clusters, nmbr of clusters = len(dct) 
                [X_train, X_test, y_train, y_test, LP_rat] = data.read_dct(dictionary, cl)   # run xgboost: train model, save model, predictions, plots 
                if len(X_train) >=800: 
                   print('Training the xgboost classification algorihtm...')                           # lower_limit=800  & upper_limit=200000 (?)
                   xgb_model.xgb_train(X_train, y_train, configs, 'gain', cl)                          # this saves the models as pickle files in saved models folder 
                   print('Done!', ' Running the prediction based on a saved model...', ' using', len(X_train), '-many data')
                   l_first, l_second = xgb_model.xgb_pred(X_test, y_test, 'gain', cl, len(X_train), LP_rat)

                   title=l_first

                   print('These are: ', l_first, l_second) 
                   print('Done!')
                                    
                   data_arr=dictionary['uni_%s'%cl]  
                   np.savetxt('./info_from_preproc/data_arr/train_array_'+str(cl)+'.txt', data_arr, newline='\n', fmt='%s')
                   data_arr=np.array(data_arr)                 # 
                   print(len(data_arr.T[l_first] ) )           #

                   l_first=data_arr.T[l_first]                 #
                   l_first=l_first.reshape((len(l_first),1))   #

                   l_end=data_arr.T[-1:]

                   print(l_first.shape)
                   print(l_end.shape)

                   hist_=np.hstack([l_first, l_end.T]) 
                   print(hist_) 

                   hist_sorted=hist_[hist_[:,0].argsort()]
                   print(hist_sorted)                   

                   c_1=[]
                   c_2=[]
                   c_3=[]
                   c_4=[]
                   c_5=[]
                   c_6=[]
                   c_7=[]
                   c_8=[]
                   c_9=[]
                   c_10=[]

                   for i in range(len(hist_sorted)): 
                       x=hist_sorted[i][0] 
                       y=hist_sorted[i][1]

                       if x>=0 and x<0.1:
                          c_1.append(y)             
          
                       elif x>=0.1 and x<0.2: 
                          c_2.append(y)

                       elif x>=0.2 and x<0.3:
                          c_3.append(y)

                       elif x>=0.3 and x<0.4: 
                          c_4.append(y)

                       elif x>=0.4 and x<0.5:
                          c_5.append(y)
          
                       elif x>=0.5 and x<0.6: 
                          c_6.append(y)

                       elif x>=0.6 and x<0.7:
                          c_7.append(y)

                       elif x>=0.7 and x<0.8: 
                          c_8.append(y)

                       elif x>=0.8 and x<0.9:
                          c_9.append(y)

                       elif x>=0.9 and x<=1.0:
                          c_10.append(y)

                   C1=np.sum(c_1)
                   C2=np.sum(c_2)
                   C3=np.sum(c_3)
                   C4=np.sum(c_4)
                   C5=np.sum(c_5)
                   C6=np.sum(c_6)
                   C7=np.sum(c_7)
                   C8=np.sum(c_8)
                   C9=np.sum(c_9)
                   C10=np.sum(c_10)

                   C=np.hstack([C1, C2, C3, C4, C5, C6, C7, C8, C9, C10])

                   plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],C)
                   plt.title('f_'+str(title))
                   plt.show()

                else:
                   print('The category with number: ', cl, 'does not have sufficient data for equal classes, skipping...') 
            else:
                print('The category with number: ', cl, 'does not have sufficient amount of rows, skipping...') 
    else:
        print('xgboost was not enabled in this run.')

#############################################################
#### Choose Use Cases and edit main() accordingly ###########
#############################################################
if __name__ == '__main__':
    main(do_preprocessing=False, do_clustering=True, do_dictionary=False, do_xgboost=True)    
 
    # Use_Case 1. To preprocess a csv file:                           do_preprocessing=True,  do_clustering=False, do_dictionary=False, do_xgboost=False
    # Use_Case 2. To create clusters on an existing'train_array.txt'  do_preprocessing=False, do_clustering=True,  do_dictionary=False, do_xgboost=False 
    # Use_Case 3: To create a dictionary on a train_array.txt file:   do_preprocessing=False, do_clustering=False, do_dictionary=True,  do_xgboost=True
    # Use_Case 4. To run xgboost on an existing dictionary:           do_preprocessing=False, do_clustering=False, do_dictionary=False, do_xgboost=True
    # Use_Case 5: To create a dictionary on given clusters:           do_preprocessing=False, do_clustering=True,  do_dictionary=True,  do_xgboost=False
    # Use_Case 6. To pre-process, cluster, dictionary and xgboost:    do_preprocessing=True,  do_clustering=True,  do_dictionary=True,  do_xgboost=True
    # Note: If do_clustering=False then all clusters are merged to 1 cluster 

#############################################################
#############################################################


