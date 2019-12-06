# import necesary python libraries
import numpy as np
import pandas as pd
#from sklearn.svm import SVR

######################################################
# import model parameters
#import params

#######################################################
# Load Walmart data - v1
#######################################################
df = pd.read_csv('./data/new_cat_data.csv')

headers=list(df.columns.values)

print(headers)

# Print out the shape of df_a
print(df.shape)

# Print out the column names in df_a
print(df.columns)

ds = df.sample(frac=0.01)


ds.to_csv('8000.csv')

print(ds.shape)

'''
# Save each one of the train data column as a separate .txt file
for i in range(df.shape[1]):
    type_= type(df.iloc[0, i])
    col=df.iloc[:, i]

    # Save in 'data_type'
    np.save('./column_data/data_type/npy/col_'+str(i) + '_' + str(type_) +'_' + str(df.columns[i]), col)
    np.savetxt('./column_data/data_type/txt/col_'+str(i) + '_' + str(type_) +'_' + str(df.columns[i]), col, fmt='%s', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

    # Save in 'column_number' 
    np.save('./column_data/col_number/npy/col_' + str(i), col)
    np.savetxt('./column_data/col_number/txt/col_' + str(i), col, fmt='%s', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

    # Save in 'feature_name' 
    np.save('./column_data/feature_name/npy/col_' + str(df.columns[i]), col)
    np.savetxt('./column_data/feature_name/txt/col_' + str(df.columns[i]), col, fmt='%s', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
   

# Create text file with column names and feature description
descr=[]

for i in range(df.shape[1]):
    type_= type(df.iloc[0, i])
    #col=df.iloc[:, i]
   
    descr.append( 'col_'+str(i) + '_' + str(type_) +'_' + str(df.columns[i]) )
    
np.savetxt('./feature_description_b', descr, fmt='%s', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)

'''








