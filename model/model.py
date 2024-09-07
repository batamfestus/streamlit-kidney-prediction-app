from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
import pickle as pickle
import pandas as pd  
import numpy as np   
from sklearn.preprocessing import StandardScaler

# Model for balancing our dataset
from imblearn.over_sampling import SMOTE

######## let us get our data
def get_data():
    data = pd.read_csv(r"C:\Users\UK-PC\Desktop\sample-project-2\KIDNEY CLASSIFICATION-2\df_model.csv")
    return data

######### let us create our model 
def create_model(data):
    X = data.drop("Diagnosis", axis=1)
    y = data['Diagnosis']

    #### let us balance our data
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    print(f"accuracy of the model: {accuracy_score(y_test, predictions)}")
    print(f"Classification Report: {classification_report(y_test, predictions)}")
    print(f"Precision Score: {precision_score(y_test, predictions)}")

    return model, scaler


def main():
    data = get_data()

    model, scaler = create_model(data)

    with open(r"C:\Users\UK-PC\Desktop\sample-project-2\KIDNEY CLASSIFICATION-2\model\model.pkl", 'wb') as f:
        pickle.dump(model, f)  
    
    with open(r"C:\Users\UK-PC\Desktop\sample-project-2\KIDNEY CLASSIFICATION-2\model\scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()