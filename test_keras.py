import numpy as np
import matplotlib.pyplot as plt

a=[ [0.1, 1], [0.15, 1], [0.25, 1], [0.35, 1], [0.45, 1], [0.55, 1], [0.65, 1], [0.1, 1], [0.15, 1], [0.25, 1], [0.35, 1], [0.45, 1], [0.55, 1], [0.65, 1], [0.1, 1], [0.15, 1], [0.25, 1], [0.35, 1], [0.45, 1], [0.55, 1], [0.65, 1], [0.1, 1], [0.15, 1], [0.25, 1], [0.35, 1], [0.45, 1], [0.55, 1], [0.65, 1] ]

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

for i in range(len(a)): 
    x=a[i][0] 
    y=a[i][1]

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

plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],C)
plt.show()









