#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score


# In[ ]:





# In[3]:


#Load Raw data from the disk
with (open("all_features.pickle", "rb")) as openfile:
    while True:
        try:
            all_features = pickle.load(openfile)
        except EOFError:
            break


# In[19]:


count  = 0
svm_input_X = []
svm_input_Y = []


for ticker in all_features.keys():
    count +=1
    for feature in all_features[ticker].keys():
        stock_df = all_features[ticker][feature]["data"]
        label = all_features[ticker][feature]["label"]
        ind = stock_df.index.get_loc(0)
        stock_target_df = stock_df.iloc[:ind] #last day of the set minus 120 days and plus 21 days
        #print(stock_target_df.index)
        #print(label)
        time_series = stock_target_df["close"].values.reshape(1,-1)
        if(label != "ambiguous" and len(time_series[0]) == 120):
           
            #print(len(time_series[0]))
            svm_input_X.append(time_series[0])
            svm_input_Y.append(label)
    if(count>4):
        break


# In[20]:



X_train, X_test, y_train, y_test = train_test_split(svm_input_X, svm_input_Y, test_size=0.2,
                                                    random_state=42)


# In[21]:


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svm = GridSearchCV(SVC(kernel='linear', gamma=0.1), cv=5,
                    param_grid=tuned_parameters,n_jobs = 6)
svm.fit(X_train,y_train)


# In[22]:


y_pred = svm.predict(X_test)
correct = accuracy_score(y_test, y_pred, normalize=False)
print(len(y_pred) == len(y_test))


# In[23]:


print("The accuracy is ",correct/len(y_test))


# In[26]:


pickle_out = open("svm_model.pickle","wb")
pickle.dump(svm, pickle_out)
pickle_out.close()


# In[ ]:





# In[ ]:




