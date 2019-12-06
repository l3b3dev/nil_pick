# import necesary python libraries
import numpy as np
import pandas as pd

print('Reading csv file...')
df = pd.read_csv('./data/new_cat_data.csv')
print('Done!')

# Print out the shape of df_a
print(df.shape)

# Print out the column names in df_a
print(df.columns)

# Issue: the above are not equal and this will not work

'''
# OR to see all columns line-by-line
for i in range(len(df.columns)):
    print(df.columns[i])

# print uniques of 'mdse_catg_nbr'
unique_cat_nbr=df['mdse_catg_nbr'].unique()

# print uniques of 'mdse_catg_desc'
unique_cat_desc=df['mdse_catg_desc'].unique()

#
print(len(unique_cat_nbr))
print(len(unique_cat_desc))

# 
mapping=[]
for i in range(len(unique_cat_nbr)):
    mapping=[unique_cat_nbr[i], unique_cat_desc[i]]

np.savetxt('./info_from_preproc/mapping.txt', mapping, newline='\n', fmt='%s' )

'''

#
df_new=df.sort_values(['mdse_catg_nbr']) 

mapping=np.vstack( [df_new['mdse_catg_nbr'], df_new['mdse_catg_desc'] ]).T
#print(mapping)
#np.savetxt('./info_from_preproc/mapping.txt', mapping, newline='\n', fmt='%s' )

short_map=[]
for i in range(len(mapping)-1):
    print(mapping[i][0], mapping[i+1][0])
    if mapping[i][0] != mapping[i+1][0]: 
         print(mapping[i])
         short_map.append(mapping[i])
    

print(short_map)
   
np.savetxt('./info_from_preproc/short_map.txt', short_map, newline='\n', fmt='%s' )    
np.save('./info_from_preproc/short_map', short_map)  












