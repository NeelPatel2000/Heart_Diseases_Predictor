# importing the necessary libraries 
import pandas as pd 
import matplotlib.pyplot as plt # for confusion matrix
import seaborn as sns # for feature importance graph
import pickle # to save the machine learning model
import random # used it to make two new feature so no need for it now
import numpy as np  
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb # importing model
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV


def clean_dataset(): # helper function for cleaning and preping dataset
    df = pd.read_csv('data/heart_disease_uci.csv')
    df = df.dropna()
    df = df.drop(['dataset'], axis=1)
    df = df.drop(['id'], axis=1) # dropping id so it can't affect my model
    df = df.rename(columns={'num': 'target', 'cp':'chest_pain', 'fbs': 'fasting_bs', 
                            'trestbps':'resting_bs', 'chol': 'cholesterol', 'restecg': 'rest_ecg',
                            'thal': 'thal_defect', 'ca': 'major_vessels' }) 
    # mapping boolean and string to integer 
    df['fasting_bs'] = df['fasting_bs'].map({True : 1, False : 0}) 
    df['sex'] = df['sex'].map({'Male' : 1, 'Female' : 0})
    df['exang'] = df['exang'].map({True : 1, False : 0})
    df['thal_defect'] = df['thal_defect'].map({'normal': 0, 'fixed defect': 1, 'reversable defect' : 2})
    df['rest_ecg'] = df['rest_ecg'].map({'normal': 0, 'st-t abnormality' : 1, 'lv hypertrophy' : 2})
    df['chest_pain'] = df['chest_pain'].map({'typical angina': 0, 'asymptomatic': 1, 'non-anginal': 2,
                                             'atypical angina': 3})
    df['slope'] = df['slope'].map({'downsloping': 0, 'flat': 1, 'upsloping': 2})
    return df   

def create_model(data):
    # making the target variable 0 == no diseases and >0 == diseases 
    y = (data['target'] > 0).astype(int)
    X = data.drop(['target'], axis = 1)
    
    # splitting the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)
    
    # loading model XGBoost Classifier with some hyperparameter
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method = 'hist',
        random_state=42,
        subsample =  0.7, 
        reg_lambda = 0.5, 
        reg_alpha = 0, 
        n_estimators = 100, 
        min_child_weight =  5, 
        max_depth = 6, 
        learning_rate = 0.2, 
        colsample_bytree = 0.7)
    
    # fitting model on the training dataset
    model.fit(X_train, y_train) 
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    cm = confusion_matrix(y_test, y_pred)
    
    # printing evaluation metrics
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('ROC-AUC: ', roc_auc_score(y_test, y_prob))
    print('Classification report: ', classification_report(y_test, y_pred))
    print('\nConfusion Matrix (Actual vs Predicted)')
    print(cm) 
    
    #  confusion matrix visualization
    
    plt.figure(figsize= (5,4))
    sns.heatmap(cm, annot= True, fmt = 'd', cmap = 'Blues', cbar = False)
    plt.title('Cofusion Matrix')    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show(block=True)
    
    # feature importance plot
    
    xgb.plot_importance(model)
    plt.title('Feature Importance')
    plt.show(block=True)
    
    return model

# passing all the helper function in the main function

def main(): 
    print('Loading and cleaning dataset....')
    df = clean_dataset()
    
    print('\n Training the model....')
    model = create_model(df)
    
# saving the trained model using pikle library so when application is deployed no need to train the model again and again     
    print('\n Saving the trained model to model/model.pkl')
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print('Model training and saving completed.')

# main function is called only when the script is run directly
if __name__ == '__main__':
    main()
    
    