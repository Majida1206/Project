import numpy as np
import pandas as pd
df3=pd.read_csv(r"C:\Users\Smile\Downloads\PhishingDataset.csv")

# Split data
X=df3.drop(['label'],axis=1)
y=df3['label']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
# Predict
y_pred_rf = rf.predict(X_test)
import pickle
# Save train model
pickle.dump(rf, open('phishing_model.pickle', 'wb'))






