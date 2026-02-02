import pandas as pd
import numpy as np
import pickle


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score



df=pd.read_csv("water_potability.csv")

print(df)

# EDA 

# missing values handel

df['ph']=df['ph'].fillna(df.groupby("Potability")['ph'].transform('median'))
df['Sulfate'] = df['Sulfate'].fillna(df.groupby('Potability')['Sulfate'].transform('median'))
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby('Potability')['Trihalomethanes'].transform('median'))

# scaling

num_pipeline=Pipeline(steps=[
    ('scaler',StandardScaler())
])


# x and y select 

x=df.drop(['Potability'],axis=1)
y=df['Potability']


num_features=x.select_dtypes(include=[np.number]).columns.tolist()

# correlation 

corr=df.corr()['Potability'].sort_values(ascending=False)

# split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)



#### XGBoost model ###

# pipeline creation

new_xgb_pipeline=ImbPipeline(
    steps=[
        ('scaler', StandardScaler()),
        ('smote',SMOTE(sampling_strategy=1.0,random_state=42)),

        ('model',XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.7,
            gamma=0.2,

            random_state=42

        ))
    ]
)


# fit model 

new_xgb_pipeline.fit(X_train,y_train)
# predict 
y_pred_xgb=new_xgb_pipeline.predict(X_test)


# evalution before thersold
accuracy=accuracy_score(y_test,y_pred_xgb)
print("----without set threshold----")
print("accuracy=",accuracy)
print(f"confusion matrix: \n")
print(confusion_matrix(y_test,y_pred_xgb))
print(f"Classification report: \n")
print(classification_report(y_test,y_pred_xgb))



# thersold set for recall increase

y_probs = new_xgb_pipeline.predict_proba(X_test)[:,1] # class-1 er probability houer possibility


thresholds=np.arange(0.3,0.6,0.01)
# print(thresholds)
f1_scores=[f1_score(y_test,(y_probs>=t).astype(int)) for t in thresholds]


best_thresholds=thresholds[np.argmax(f1_scores)]
print("best thresholds=",best_thresholds)

# print(y_probs)



y_pred_tuned = (y_probs >= best_thresholds).astype(int)


# evalution after thersold 
print("----with set threshold----")
print("accuracy=",accuracy_score(y_test, y_pred_tuned))
print("confusion matrix: \n",confusion_matrix(y_test, y_pred_tuned))
print("classification report: \n ",classification_report(y_test, y_pred_tuned))





# download model 

with open('water_potability_predict.pkl','wb') as f:
  pickle.dump(new_xgb_pipeline,f)













