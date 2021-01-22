# Import libraries
import argparse
import json
import joblib
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set regularization parameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
print("Loading Data...")

dataset = run.input_datasets['flight_delays_data'].to_pandas_dataframe().dropna() # Get the training data from the estimator input

# Remove target leaker and features that are not useful
target_leakers = ['DepDel15','ArrDelay','Cancelled','Year']
dataset.drop(columns=target_leakers, axis=1, inplace=True)

# convert some variables to categorical features
columns_as_categorical = ['OriginAirportID','DestAirportID','ArrDel15']
dataset[columns_as_categorical] = dataset[columns_as_categorical].astype('object')

categorical_feature_mask = dataset.dtypes == object 
categorical_cols = dataset.columns[categorical_feature_mask].tolist()

le = LabelEncoder()
dataset[categorical_cols] = dataset[categorical_cols].apply(lambda col:le.fit_transform(col))
dataset = dataset.dropna()

# Separate features and labels
X, y = dataset[['Month', 'DayofMonth', 'DayOfWeek', 'Carrier', 'OriginAirportID', 'DestAirportID', 'CRSDepTime', 'DepDelay', 'CRSArrTime']].values, dataset['ArrDel15'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))


# work on confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat)
cnf_matrix_list = cnf_matrix.tolist()
cnf_matrix_json_ = json.dumps(cnf_matrix_list)
run.log_confusion_matrix(name='Confusion_matrix', value=cnf_matrix_json_)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
run.log_image(name='confusion_matrix_img', plot=fig)

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/flight_delays_model.pkl')

run.complete()
