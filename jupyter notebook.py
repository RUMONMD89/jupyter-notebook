#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score


# In[11]:


iris = load_iris()
print(iris.DESCR)
x = iris.data
y = iris.target


# In[15]:


clf = DecisionTreeClassifier(max_depth=3)
print(clf)
scores = cross_val_score(clf, x, y, cv=10)
print("Accuracy:%0.2f (+/- %0.2f)" %(scores.mean(),scores.std()*2))


# In[16]:


clf.fit(x,y)
plot_tree(clf, filled=True, feature_names=iris.feature_names,class_names=iris.target_names)
plt.show()


# In[ ]:




