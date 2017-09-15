KNN or "simple majority vote principle"

KNN needs 4 things:
1)Define what distance means (straight line or euclidian distance?)
-Euclidian is scikit learn's default
2)How many neighbors to look at
-5
3)Optional weighting function on the neighbor points
-ignored in this case
4)method for aggregating the classes of neighbor points
-majority vote

Warning: Didn't consider bias/variance principle