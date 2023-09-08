#!/usr/bin/env python
# coding: utf-8

# ## 1&2. Setup and Data Preparation
# #### In this section, we are going to install the required libraries and read the dataset file. Then an overview of the dataset will be shown.

# In[50]:


import pandas as pd

df = pd.read_csv('Tweets/tweets.csv')
df


# ## 3. Data Exploration
# #### In this part, we are going to see some basic statistics of our dataset

# #### The describe function will show us the mean, std, min, etc. attributes of the numerical columns. In this case there would be the id and target columns. Since the id gives us no information, we can only consider the target column. As we can see, there are 11370 sample datas with the values of 1 or 0. Since the mean of the targets is 0.18, we can understand that most of the targets are equal to 0.

# In[51]:


df.describe()


# ### Now we are going to create plots to have a visual understanding of the distribution of our data in different columns

# #### As mentioned above, most of our data targets are equal to 0

# In[41]:


import matplotlib.pyplot as plt

df['target'].value_counts().plot(kind='bar')
plt.title('Distribution of Target')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()


# #### The following plot shows the distribution of different keywords in our dataset.
# #### As we can see, there are some keywords that have been appeared only once, and the maximum datas with similar keyword in their text description is around 90.

# In[42]:


df['keyword'].value_counts().plot(kind='barh')


# #### In order to find the Nan values, we can use the isna() function. As we can see, there are 3418 Nan values in the location attribute of our dataset.

# In[43]:


df.isna().sum()


# ## 4. Data Preprocessing
# #### In this section, we are going to perform some preprocessing before training our ML model. 

# #### Instead of complex text analysis on the "text" attribute of our dataset, we ae going to add a new attribute which is text size to our dataset columns.

# In[52]:


sen_lengths = []
for sentence in df['text']:
    sen_lengths.append(len(sentence.split()))
df = df.drop('text', axis=1)
df['text_size'] = sen_lengths
df


# #### Just to have an understanding of the distribution of text lengths, we show a plot that shows most of our datas have the length of 21. Also the describe function gives us a sense of our new attribute.

# In[53]:


df['text_size'].value_counts().plot(kind='bar')


# In[54]:


df.describe()


# #### Now we are going to drop the rows with a Nan value in their location. Our dataset's size will decrease to 7952.

# In[56]:


new_df = df.dropna()
new_df


# #### Since "Keyword" and "location" are categorical variables, we would encode them to numbers so that our ML model would have a better sense of the data and predict the target value better.

# In[57]:


new_df.keyword = pd.Categorical(new_df.keyword)
new_df['keyword'] = pd.factorize(new_df['keyword'])[0] + 1
new_df.location = pd.Categorical(new_df.location)
new_df['location'] = pd.factorize(new_df['location'])[0] + 1
new_df


# ## 5&6. Model building and training

# #### Now we are going to create our train and test datasets and choose a model for training
# #### The test dataset would have the size of %20 and the train dataset would have the size of %80.

# In[58]:


from sklearn.model_selection import train_test_split

y = new_df['target']
X = new_df.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# #### Now that we have our train and test datasets, we are going to train our model

# ## 7. Evaluation
# #### The accuracy function is used to calculate the accuracy of our final prediction.

# In[86]:


from sklearn import metrics

def accuracy(ans, predicted):
    sum = 0
    n = len(predicted)
    count= 0
    for y in ans:
        temp = 0
        if predicted[count] < 0.5:
            temp = 0
        else:
            temp = 1
        if temp == y:
            sum += 1
        count += 1
    return sum / n


# In[79]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Accuracy: ', accuracy(y_test, y_pred))


# #### Now we change our algorithm to decision tree classification

# In[83]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy: ', accuracy(y_test, y_pred))


# #### As we can see, the accuracy dropped after using the decision tree clasifier
# #### Now we are going to change the model to KNN

# In[87]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy: ', accuracy(y_test, y_pred))


# ## 8.Hyperparameter tuning
# #### changing the K to 3

# In[89]:


k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy: ', accuracy(y_test, y_pred))


# #### changing the K to 13

# In[103]:


k = 13
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy: ', accuracy(y_test, y_pred))


# #### Now that we have found the best accuracy, we will show the confusion matrix and classification report

# In[104]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generate and print classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# ### To create a machine learning model, there are some fundamental steps that we should take. As per this assignment, we first took a general look at our dataset and saw how the data was distributed. Then by preprocessing the dataset and cleaning the unnecessary invalid data, we made a better-understanding dataset to our machine learning model.
# ### Finally, by trying different models for training, we found out the best model to train for our dataset and showed the accuracy of its prediction.
# ### This assignment was a great experience for reviewing the basics of creating any ML model. 
