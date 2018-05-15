#Brain State Classification using Machine Learning Models

SVM

1. Brain Mask Threshold 300 + SVM (c=1, gamma=1/n_features) + PCA 50 n_components gave an accuracy ~ 45%.
2. Brain Mask Threshold 300 + SVM (c=1, gamma=1/n_features) + PCA 92 n_components gave an accuracy ~ 57%.
3. Brain Mask Threshold 300 + SVM (c=100, gamma=0.01) + PCA 92 n_components gave an accuracy ~ 59%.
4. Brain Mask Threshold 100 + SVM (c=1, gamma=1/n_features) + PCA 92 n_components gave an accuracy ~ 50%.
5. Brain Mask Threshold 300 + SVM (c=1, gamma=1/n_features) + no PCA ~ 45%.

Gradient Boosting Classifier

1. Brain Mask Threshold 300 + GBC (n_estimators = 100) + PCA 50 n_components gave an accuracy ~ 59%.
2. Brain Mask Threshold 300 + GBC (n_estimators = 100) + PCA 92 n_components gave an accuracy ~ 72%.
3. Brain Mask Threshold 300 + GBC (n_estimators = 500) + PCA 92 n_components gave an accuracy ~ 77%.
4. Brain Mask Threshold 100 + GBC (n_estimators = 100) + PCA 92 n_components gave an accuracy ~ 62%.
5. Brain Mask Threshold 300 + GBC (n_estimators = 100) + no PCA ~ 80%.

Best Trail: Brain Mask Threshold 300 + SVM (c=1, gamma=1/n_features) + no PCA ~ 80%.
