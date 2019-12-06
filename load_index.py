
import numpy as np
   
map_=np.load('./info_from_preproc/short_map.npy', allow_pickle=True)  

a=np.where(map_==1018)[0][0]
print(map_[a][1])

#print(map_[10])
