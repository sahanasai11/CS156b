
# coding: utf-8

# In[16]:


import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('/groups/CS156b/data/student_labels/train.csv')
df2 = pd.read_csv('/groups/CS156b/data/student_labels/test_ids.csv')
#df.dropna()


# In[2]:


#df = df[(df['Sex'] != 'Unknown')]
df.columns


# In[3]:


sex = df['Sex']
age = df['Age']
sex_dict = {
    'Female' : 0,
    'Male' : 1, 
    'Unknown' : 2
}
frontal_lateral = {
    'Frontal' : 0,
    'Lateral' : 1
}
pa = {
    'PA' : 0,
    'AP' : 1
}
df['Frontal/Lateral'] = df['Frontal/Lateral'].replace(frontal_lateral).astype(int)
df['Sex'] = df['Sex'].replace(sex_dict).astype(int)
#df['AP/PA'] = df['AP/PA'].replace(pa).astype(int)


# In[4]:


vec = []
labels = []
for index, row in df.iterrows():
    temp = [row['Sex'], row['Age'], row['Frontal/Lateral']]
    temp2 = [row['No Finding'], row['Enlarged Cardiomediastinum'], row['Cardiomegaly'],
            row['Lung Opacity'], row['Lung Lesion'], row['Edema'], row['Consolidation'],
            row['Pneumonia'], row['Atelectasis'], row['Pneumothorax'], row['Pleural Effusion'], 
            row['Pleural Other'], row['Fracture'], row['Support Devices']]
    i = 0 
    for val in temp2: 
        if val != val:
            temp2[i] = 0.0
        i += 1
    vec.append(temp)
    labels.append(temp2)
    
labels


# In[7]:


from sklearn.multioutput import MultiOutputClassifier
X_train, X_test, y_train, y_test = train_test_split(
    vec,
    labels,
    test_size=0.3,
    random_state=42,
)

clf = tree.DecisionTreeClassifier()
clf = MultiOutputClassifier(clf, n_jobs=-1)
clf.fit(X=X_train, y=y_train)
#clf.feature_importances_ 
#clf.score(X=X_test, y=y_test) # 1.0
predictions=clf.predict(X_test) 


# In[33]:


labels[0][0] == labels[0][0]


# In[9]:


np.savetxt("test_sub.csv", predictions, delimiter=",")


# In[17]:


df2

