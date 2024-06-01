import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

df= pd.read_csv('creditcard.csv')
print(df.head())
print(df.info())
sns.countplot(df['Class'])
plt.title('Sinif Dagilimi')
plt.show()

df.drop('Class', axis=1).hist(figsize=(20,20))
plt.show()
print(df.isnull().sum())

x = df.drop('Class', axis=1)
y = df['Class']
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

sm= SMOTE(random_state= 42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

model = RandomForestClassifier(random_state=42)
model.fit(x_train_res, y_train_res)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')     
plt.xlabel('Tahmin edilen')
plt.ylabel('Gercek')
plt.title('Karisiklik Matrisi')            
plt.show()

y_pred_prob = model.predict_log_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color= 'darkorange', lw=2,label=f'ROC egrisi(AUC= {roc_auc:0.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Yanlis Pozitif Orani')
plt.ylabel('Dogru Pozitif Orani')
plt.title('Alici İsletim Karakteristigi (ROC) Egrisi')
plt.legend(loc='lower right')
plt.show()

param_grid= {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy']
} 

grid_search= GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(x_train_res, y_train_res)
print(f'En iyi parametreler: {grid_search.best_params_}')
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(x_test)
print(classification_report(y_test, y_pred_best))

conf_matrix_best = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin edilen')
plt.ylabel('Gercek')
plt.title('Karisiklik Matrisi (Iyileştirilmiş Model)')            
plt.show()