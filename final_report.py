import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

#Insert the database
df=pd.read_csv('data/election2.csv')

#Get the labels from the csv file
labels=df.value

#Label the stance according to the values,
#higher than 0 = support
#equal to 0 = neutral
#lower to 0 = against
conditions = [
    (df['value'] < 0),
    (df['value'] == 0),
    (df['value'] > 0)
    ]

values = ['against', 'neutral', 'support']
df['stand'] = np.select(conditions, values)
#combine the rows into one entity
df["policy"] = df["title"] + df["party"] + df["view"]
df["policy"].head()

#Split the dataset according to the proportion of the test set
x_train,x_test,y_train,y_test=train_test_split(df['policy'], labels.astype('int'), test_size=0.3, random_state=7)

#Initialize a TfidfVectorizer with a threshold of 70% of the document in here
tv=TfidfVectorizer(stop_words='english', max_df=0.7)
#Fit and transform train set, transform test set
t_train=tv.fit_transform(x_train) 
t_test=tv.transform(x_test)

#Initialize a PassiveAggressiveClassifier for predictions
pac=PassiveAggressiveClassifier(max_iter=500)
pac.fit(t_train,y_train)
#Predict on the test set and calculate accuracy
y_pred=pac.predict(t_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
