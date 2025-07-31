# ML_Yemekhane
## Please install the following packages before running the code
```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install scipy
pip install xgboost
```
Note: Xgboost is used in the RandomForest.ipynb file. If you do not want to use it, you can comment out the import statement in the RandomForest.ipynb file. We are using that to see what is the best result we can get.

## Replication steps
- Please locate the data "dataset_new_cat.csv" under prepare_dataset/data folder.
- To get the results for the k-means clustering please run all the cells in the kmeans.ipynb file under the Code/KMeans folder.
- To get the results for logistic regression, run the code under./to_submit/myLogisticRegression.py for my implementation. (you can open the code and edit to switch between SimpleLogisticRegressionModel (no regularization) and LogisticRegressionModel (has regularization and early stopping)). For built in implementation run the code in
./to_submit/Logistic_Regression_builtin.py

- To get the results for the random forests, please run all the cells in the RandomForest.ipynb file under the Code/RandomForest folder.

- To get the results for the k-Nearest Neighbors, run the code under ./to_submit/knn_model.py
  
- To run logistic regression run the following command (for mymyLogisticRegression.py you can switch between simple)
```bash
python ./to_submit/myLogisticRegression.py
python ./to_submit/Logistic_Regression_builtin.py
```

- To run the svm implementation run the following command
```bash
python ./SVM/svm_all.py
```

- To run decision boundary run the following command
```bash
python ./DecisionBoundary/draw_dec_bound.py
```
- To run kNN run the following command
```bash
python ./to_submit/knn_model.py
```

