import numpy as np
import pandas as pd
from scipy import io
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# SES-TEST DATA

all_train_data = io.loadmat('sub-01/ses-test/func/fMRImask.mat')
labels = io.loadmat('label.mat')['label']
train_data = all_train_data['fMRIdata_1D_ROI']
print("Training Dataset Shape", train_data.shape)
print("Label Shape", labels.shape)

X = pd.DataFrame(train_data)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5)

svc_1 = SVC()
svc_1.fit(X_train, y_train)
y_pred_svc = svc_1.predict(X_test)

print("The accuracy of the model Support Vector Machine is: ", accuracy_score(y_test, y_pred_svc))

gbc_1 = GradientBoostingClassifier()

gbc_1.fit(X_train, y_train)
y_pred_gbc = gbc_1.predict(X_test)

print("The accuracy of the model Gradient Boosting Classifer is: ", accuracy_score(y_test, y_pred_gbc))

gbc_b = GradientBoostingClassifier(n_estimators=500)

gbc_b.fit(X_train, y_train)
y_pred_gbc_b = gbc_b.predict(X_test)

print("The accuracy of the model Gradient Boosting Classifer with tuned parameters is: ", accuracy_score(y_test, y_pred_gbc_b))

# Model Training with PCA

pca = PCA()
pca.fit(X_train, y_train)
pca_train = pca.transform(X_train)
pca_test = pca.transform(X_test)

svc_2 = SVC()
svc_2.fit(pca_train, y_train)
y_pred_svc_pca = svc_2.predict(pca_test)

print("The accuracy of the model Support Vector Machine with PCA is: ", accuracy_score(y_test, y_pred_svc_pca))

gbc_2 = GradientBoostingClassifier()

gbc_2.fit(pca_train, y_train)
y_pred_gbc_pca = gbc_2.predict(pca_test)

print("The accuracy of the model Gradient Boosting Classifer with PCA is: ", accuracy_score(y_test, y_pred_gbc_pca))

gbc_b_2 = GradientBoostingClassifier(n_estimators=500)

gbc_b_2.fit(X_train, y_train)
y_pred_gbc_pca_b_2 = gbc_b_2.predict(X_test)

print("The accuracy of the model Gradient Boosting Classifer with PCA and tuned parameters is: ", accuracy_score(y_test, y_pred_gbc_pca_b_2))

# SES-RETEST DATA

all_train_data = io.loadmat('sub-01/ses-retest/func/fMRImask.mat')
labels = io.loadmat('label.mat')['label']
train_data = all_train_data['fMRIdata_1D_ROI']
print("Training Dataset Shape", train_data.shape)
print("Label Shape", labels.shape)

X = pd.DataFrame(train_data)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5)

svc_retest_1 = SVC()
svc_retest_1.fit(X_train, y_train)
y_pred_svc_retest = svc_retest_1.predict(X_test)

print("The accuracy of the model Support Vector Machine is: ", accuracy_score(y_test, y_pred_svc_retest))

gbc_retest_1 = GradientBoostingClassifier()

gbc_retest_1.fit(X_train, y_train)
y_pred_gbc_retest = gbc_retest_1.predict(X_test)

print("The accuracy of the model Gradient Boosting Classifer is: ", accuracy_score(y_test, y_pred_gbc_retest))

gbc_retest_b = GradientBoostingClassifier(n_estimators=500)

gbc_retest_b.fit(X_train, y_train)
y_pred_gbc_retest_b = gbc_retest_b.predict(X_test)

print("The accuracy of the model Gradient Boosting Classifer is: ", accuracy_score(y_test, y_pred_gbc_retest_b))

# Model Training with PCA

pca = PCA()
pca.fit(X_train, y_train)
pca_train = pca.transform(X_train)
pca_test = pca.transform(X_test)

svc_2_retest = SVC()
svc_2_retest.fit(pca_train, y_train)
y_pred_svc_pca_retest = svc_2_retest.predict(pca_test)

print("The accuracy of the model Support Vector Machine with PCA is: ", accuracy_score(y_test, y_pred_svc_pca_retest))

gbc_2_retest = GradientBoostingClassifier()

gbc_2_retest.fit(pca_train, y_train)
y_pred_gbc_pca_retest = gbc_2_retest.predict(pca_test)

print("The accuracy of the model Gradient Boosting Classifer with PCA is: ", accuracy_score(y_test, y_pred_gbc_pca_retest))

gbc_b_retest = GradientBoostingClassifier(n_estimators=500)

gbc_b_retest.fit(X_train, y_train)
y_pred_gbc_b_retest = gbc_b_retest.predict(X_test)

print("The accuracy of the model Gradient Boosting Classifer with tuned parameters is: ", accuracy_score(y_test, y_pred_gbc_b_retest))
