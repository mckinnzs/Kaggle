# Kaggle
kaggle competition codes

This project contains my entry into the Kaggle Ghoul competition.

Models explored include classifiers, namely RandomForestClassifier(n_estimators=20) and GaussianNB(), and ensemble classifiers,
i.e., VotingClassifier(estimators=[('rf', clf1), ('nb', clf2)], voting='hard'), BaggingClassifier(AdaBoostClassifier(clf1, n_estimators=50,
 algorithm='SAMME.R')), AdaBoostClassifier(clf1, n_estimators = 50, algorithm='SAMME.R'), and GridSearchCV(eclf3, param_grid=param_grid).
