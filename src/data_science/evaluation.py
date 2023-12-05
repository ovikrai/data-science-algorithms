from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def show_confusion_matrix(y_test, prediction):
    cm = confusion_matrix(y_test, prediction)
    print('----------------CONFUSION MATRIX-----------------------------')
    print('-------------------------------------------------------------')
    print('True  Positive: ', cm[0][0], 'False Negatives: ', cm[0][1])
    print('False Positive: ', cm[1][0], 'True  Negatives: ', cm[1][1])
    print('-------------------------------------------------------------')


def classifier_report(y_test, prediction):
    print('----------------CLASSIFICATION REPORT------------------------')
    print(classification_report(y_test, prediction))
    print('-------------------------------------------------------------')


def grid_search(x_train, y_train, classifier, parameters,
                scoring='accuracy', cv=10, n_jobs=-1):
    gs = GridSearchCV(estimator=classifier,
                      param_grid=parameters,
                      scoring=scoring,
                      cv=cv,
                      n_jobs=n_jobs)
    gs.fit(x_train, y_train)
    best_accuracy = gs.best_score_
    best_parameters = gs.best_params_
    print('----------------GRID SEARCH----------------------------------')
    print('Best Accuracy:   ', best_accuracy * 100)
    print('Best Parameters: ', best_parameters)
    print('-------------------------------------------------------------')


def k_fold_cross_validation(classifier, x_train, y_train, cv=10):
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=cv)
    print('----------------CROSS VALIDATION-----------------------------')
    print('Accuracy:           ', accuracies.mean() * 100)
    print('Standard Deviation: ', accuracies.std() * 100)
    print('-------------------------------------------------------------')
