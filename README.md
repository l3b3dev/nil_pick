# nil_pick

We will try to address the nil picking problem in this project. This problem is very common for online ordering e-commerce platforms like Amazon, Walmart or Fanatics. The nil pick occurs when one or more orders are received for a product and when an insufficient amount of the product exists at the warehouse to satisfy the one or more orders.  The problem also exists if there is a human fulfilling the order in the warehouse and he cannot locate the product, but the product exists or is misplaced. When the order comes in it has a lot of data related to it and the main idea is to be able to predict for each row of orders if this row is going to be nil picked. We also would like to know which features that were used for the decision making are important and which are not

* To Run modify [Run program](run.py) for the following use cases.  Default is Use_Case 4
```python
   # Use_Case 1. To preprocess a csv file:                            do_preprocessing=True,  do_clustering=False, do_dictionary=False, do_xgboost=False
   # Use_Case 2. To create clusters on an existing'train_array.txt'  do_preprocessing=False, do_clustering=True,  do_dictionary=False, do_xgboost=False 
   # Use_Case 3. To run xgboost on an existing dictionary:           do_preprocessing=False, do_clustering=True, do_dictionary=False, do_xgboost=True
   # Use_Case 4. To pre-process, cluster, dictionary and xgboost:    do_preprocessing=True,  do_clustering=True,  do_dictionary=True,  do_xgboost=True
```

*Preprocessing

*Clustering
 We first need to preprocess the data by configuring the program in run.py with do_preprocessing=True, do_clustering=False, do_dictionary=False, do_xgboost=False
This will rely on our DataLoader class which has methods to preprocess_numerical, preprocess_categorical, preprocess_target and methods related to creating cluster dictionary. Those methods mostly involve data cleaning/normalization as well as dumping statistics to sanity_checks folder. The main output of those methods are numberical_data, categorical_data and target arrays which will get np.hstack into train_array which will be saved in ./info_from_preproc/train_array.txt  

*Dictionary Creation

*XGBoost

*Parameters finetuning