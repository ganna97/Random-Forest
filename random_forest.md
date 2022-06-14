```python
# Importing the required libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline
```


```python
df= pd.read_stata("C:\\Users\\Admin\\Desktop\\NSSOData\\inpatients_without_deliveries")
```


```python
data= df.drop(['centre', 'fsu', 'round', 'schedule', 'sample',
       'district', 'stratum', 'sub_stratum', 'sub_round', 'sub_sample', 'fod', 'hg_sb', 'sss',
       'hh', 'level', 'filler',  'informant_srno', 'response_code', 'survey_code',
       'substitution_code', 'employee_codea', 'employee_codeaa',
       'employedd_codeb', 'survey_date', 'despatch_date', 'canvass', 'investigator_no',
       'remark_a', 'remark_b', 'remark_elsea', 'remark_elseb', 'nss', 'nsc', 'mlt', 'nic_2008',
       'nco_2004'], axis= "columns")
```


```python
# Putting feature variable to X
X = df.drop('',axis=1)
# Putting response variable to y
y = df['CATAS']
```


```python
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
#pd.set_option("display.height", 1000)
pd.set_option("display.width", 1000)
```


```python
data.head()
```


```python
data.columns
```


```python
X= data[['CATAS','dis_m','Dis_cat','agegroups', 'religion1', 'Community', 'Education', 'Gender', 'Type_of_Insurance', 'type_of_Hospital', 'sector', 'region', 'hhsize','Mstatus', 'Hospital_Type', 'head_relation', 'work_activity', 'HH_size', 'mpce2_level', 'Hospital_duration', 'DHF','death_status', 'ypce']]

```


```python
y = X['CATAS']
X= X.drop(['CATAS'], axis= 'columns')
```


```python
from sklearn.preprocessing import LabelEncoder
```


```python
le= LabelEncoder()
X['Gender']= le.fit_transform(X['Gender'])
```


```python
X['Community']= le.fit_transform(X['Community'])
```


```python
X['dis_m']= le.fit_transform(X['dis_m'])
```


```python
X['Dis_cat']= le.fit_transform(X['Dis_cat'])
```


```python
X['agegroups']= le.fit_transform(X['agegroups'])
```


```python
X['religion1']= le.fit_transform(X['religion1'])
```


```python
X['Education']= le.fit_transform(X['Education'])
```


```python
X['Type_of_Insurance']= le.fit_transform(X['Type_of_Insurance'])
```


```python
X['type_of_Hospital']= le.fit_transform(X['type_of_Hospital'])
X['mpce2_level']= le.fit_transform(X['mpce2_level'])
X['region']= le.fit_transform(X['region'])
X['sector']= le.fit_transform(X['sector'])
X['head_relation']= le.fit_transform(X['head_relation'])
X['work_activity']= le.fit_transform(X['work_activity'])
X['Hospital_duration']= le.fit_transform(X['Hospital_duration'])
X['death_status']= le.fit_transform(X['death_status'])
X['DHF']= le.fit_transform(X['DHF'])
```


```python
X['Mstatus']= le.fit_transform(X['Mstatus'])
X['ypce']= le.fit_transform(X['ypce'])
X['HH_size']= le.fit_transform(X['HH_size'])
```


```python
X.head()
```


```python
X.columns
```


```python
X= X.drop(['Hospital_Type'], axis= "columns")
```


```python
X.type_of_Hospital
```


```python
df.Gender.unique()
```


```python
# now lets split the data into train and test
from sklearn.model_selection import train_test_split
```


```python
X.shape, y.shape
```


```python
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

X_train.shape, X_test.shape
```


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)
```


```python
%%time
classifier_rf.fit(X_train, y_train)
```


```python
# checking the oob score
classifier_rf.oob_score_
```


```python
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
```


```python
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}
```


```python
from sklearn.model_selection import GridSearchCV
```


```python
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
```


```python
%%time
grid_search.fit(X_train, y_train)
```


```python
grid_search.best_score_
```


```python
rf_best = grid_search.best_estimator_
rf_best
```

#  To visualize the Trees


```python
from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[5], feature_names = X.columns,class_names=['Disease', "No Disease"],filled=True);
```


```python
from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators_[7], feature_names = X.columns,class_names=['Disease', "No Disease"],filled=True);
```


```python
rf_best.feature_importances_
```


```python
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})
```


```python
imp_df[imp_df.Imp<0.02].Varname
```


```python
X1= X.drop([ 'religion1', 'Education', 'Gender', 'sector', 'Mstatus', 'head_relation', 'work_activity', 'HH_size', 'DHF', 'death_status'], axis= "columns")
```


```python
# Splitting the data into train and test
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, train_size=0.7, random_state=40)

X1_train.shape, X1_test.shape
```


```python
X1.shape, y.shape
```


```python
classifier_rf = RandomForestClassifier(random_state=40, n_jobs=-1, max_depth=5,
                                       n_estimators=25, oob_score=True)
```


```python
%%time
classifier_rf.fit(X1_train, y_train)
```


```python
# checking the oob score
classifier_rf.oob_score_
```


```python
rf = RandomForestClassifier(random_state=40, n_jobs=-1)
```


```python
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}
```


```python
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
```


```python
%%time
grid_search.fit(X1_train, y_train)
```


```python
grid_search.best_score_
```


```python
rf_best = grid_search.best_estimator_
rf_best
```


```python
plt.figure(figsize=(180,40))
plot_tree(rf_best.estimators_[3], feature_names = X1.columns,class_names=['Incurred', "Not Incurred"],filled=True);
```


```python
rf_best.feature_importances_, X1.columns
```


```python
imp_df = pd.DataFrame({
    "Varname": X1_train.columns,
    "Imp": rf_best.feature_importances_
})
```


```python
imp_df
```


```python
X1.dtypes
```


```python
X1['mpce2_level'] =X1['mpce2_level'].astype('category')
```


```python
X1= X1.drop(['region '], axis='columns')
```


```python
X1.mpce2_level
```


```python
X2.columns 
```


```python
x.dtypes
```


```python
print(x.loc[x['hhsize']>25 , ['ypce', 'sector', 'hhsize', 'ypce']])
```


```python
# to use creating a variable using existing
f= lambda x: "low ypce" if x<5000 else "high ypce"
x['wealth']= x.ypce.apply(f)
print(x[['ypce','wealth']])
```


```python
X.HH_size.max()
```


```python
x.hhsize
```


```python
print(na_)
```


```python
X= X.dropna(subset=['Type_of_Insurance'])
```


```python
X.shape
```
