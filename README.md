# nil_pick

We will try to address the nil picking problem in this project. This problem is very common for online ordering e-commerce platforms like Amazon, Walmart or Fanatics. The nil pick occurs when one or more orders are received for a product and when an insufficient amount of the product exists at the warehouse to satisfy the one or more orders.  The problem also exists if there is a human fulfilling the order in the warehouse and he cannot locate the product, but the product exists or is misplaced. When the order comes in it has a lot of data related to it and the main idea is to be able to predict for each row of orders if this row is going to be nil picked. We also would like to know which features that were used for the decision making are important and which are not
![Image of Yaktocat](https://github.com/lightningorders/nilpick.png)
* To Run modify [Run program](run.py) for the following use cases.  Default is Use_Case 4
```python
   # Use_Case 1. To preprocess a csv file:                            do_preprocessing=True,  do_clustering=False, do_dictionary=False, do_xgboost=False
   # Use_Case 2. To create clusters on an existing'train_array.txt'  do_preprocessing=False, do_clustering=True,  do_dictionary=False, do_xgboost=False 
   # Use_Case 3. To run xgboost on an existing dictionary:           do_preprocessing=False, do_clustering=True, do_dictionary=False, do_xgboost=True
   # Use_Case 4. To pre-process, cluster, dictionary and xgboost:    do_preprocessing=True,  do_clustering=True,  do_dictionary=True,  do_xgboost=True
```

* Preprocessing
    * We first need to preprocess the data by configuring the program in [Run program](run.py) with _do_preprocessing=True, do_clustering=False, do_dictionary=False, do_xgboost=False_
This will rely on our [DataLoader](./core/data_processor.py)  class which has methods to _preprocess_numerical, preprocess_categorical, preprocess_target_ and methods related to creating cluster dictionary. Those methods mostly involve data cleaning/normalization as well as dumping statistics to _sanity_checks_ folder. The main output of those methods are _numberical_data, categorical_data_ and _target_ arrays which will get _np.hstack_ into _train_array_ which will be saved in _./info_from_preproc/train_array.txt_  

* Clustering
    * The clustering is done by configuring the program in [Run program](run.py) with
_do_preprocessing=False, do_clustering=True, do_dictionary=True, do_xgboost=False_
Its implemented in [ClusterModel](./core/cluster_model.py). This attempts to run KMeans clustering to have balanced classes in each category. We attempt cluster sizes from 1 to 21 then use Elbow Method for optimum clusters. Our default clusters are just grouping by category number (_mdse_catg_nbr_). And as it turned out from many runs, KMeans were not very meaningful here and actually grouping by category number was always a better choice.
 
* Dictionary Creation
    * Once the clustering is decided for convenience we load _train_array_ into memory in the form of a dictionary by delegating to [DataLoader](./core/data_processor.py) class _create_train_dct_ method. It will initialize the key for each cluster in the dictionary and loop over all data rows of feature _“mdse_cat_nbr_” and append ith entry to the key, specific to the category-cluster.
    
* XGBoost
    * Running XGBoost is done by configuring the program in [Run program](run.py) with
_do_preprocessing=False, do_clustering=True, do_dictionary=False, do_xgboost=True_
The work is performed by [XGB_Model](./core/xgb_model.py) class which contains _xgb_train_ and _xgb_pred_ methods. In our training we create the Xgboost specific DMatrix data format from our existing numpy array. Then we set xgboost params (maximum depth of tree, training steps, number of classes) finally after running the training we store the model in _./saved_models/cluster_{clusterID}_. In prediction we load the model and extract most confident predictions, get importance scores based on _weight, gain, cover, total_gain_ and _total_cover_. 

* Parameters finetuning
    * We repeatedly run XGBoost using gbtree as a booster while varying the following hyper parameters of XGBoost: 
_max_depth, eta, gamma, min_child_weight, max_delta_step, subsample, and L1 regularization (alpha), L2 regularization (lambda)_. We evaluated results by running _GridSearchCV_ method. Empirically, we found that we can achieve over 90% desired accuracy by using _max_depth=8, eta=0.3, gamma=0, min_child_weight=3,max_delta_step=1_ and taking defaults for the rest of params.
