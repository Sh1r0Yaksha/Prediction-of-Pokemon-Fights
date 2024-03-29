import pandas as pd
from IPython.display import display

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

import os
parent_dir = os.path.dirname(os.getcwd())

############################       Preparing training dataset      #####################################


# Importing the preprocessed data
pokemon = pd.read_csv(parent_dir + '/data/preprocessed.csv')

# Removing the extra column at the end of the csv 
pokemon.drop(pokemon.columns[-1], axis=1)

# Defining functions for getting features from the pokemon dataset according to the index of the Pokemon and a function for renaming the columns

# Function for getting all of the features of a pokemon given their 'id'
def get_features(id,pokemon):
    return (pokemon.iloc[id - 1, 0:-1:1])

# first_or_second is a bool argument, passing True will return first pokemon and False will return second pokemon
def rename_column(first_or_second, features):
     list_ = features.columns.to_list()
     if first_or_second:
          for i in range(len(list_)):
               list_[i] = 'First_' + list_[i] 

     else:
          for i in range(len(list_)):
               list_[i] = 'Second_' + list_[i] 

     features.columns = list_

# Importing training data
combats = pd.read_csv(parent_dir + '/data/combats.csv')

# Creating a dataframe which will have all the features of the 'First_Pokemon' from 'combats.csv' with column name renamed
first_pokemon = pokemon.copy()
rename_column(True, first_pokemon)

list_first = combats.iloc[:, 0].to_list()
temp_ = list()

for i in range(len(list_first)):
    temp_.append(get_features(list_first[i], first_pokemon).to_frame().T)

df_first_pokemon = pd.concat(temp_, ignore_index=True)

# Similar as above step, dataframe for 'Second_Pokemon'
temp_.clear()
second_pokemon = pokemon.copy()
rename_column(False, second_pokemon)

list_second = combats.iloc[:, 1].to_list()


for i in range(len(list_second)):
    temp_.append(get_features(list_second[i], second_pokemon).to_frame().T)

df_second_pokemon = pd.concat(temp_, ignore_index=True)

# Removing Pokemon ID feature which is unnecessary for training
df_first_pokemon = df_first_pokemon.drop([df_first_pokemon.columns[0],df_first_pokemon.columns[1]], axis=1)
df_second_pokemon = df_second_pokemon.drop([df_second_pokemon.columns[0], df_second_pokemon.columns[1]], axis=1)

# Joining the first and second pokemon dataframe to create the final training dataframe
training_df = pd.concat([df_first_pokemon, df_second_pokemon], axis=1)

# Creating class labels: 1 means first Pokemon wins, 0 means second Pokemon wins
class_labels = []

x = combats.iloc[:,2].to_list()

for i in range(len(x)):
    if (x[i] == list_first[i]):
        class_labels.append(1)
    else:
        class_labels.append(0)


################################# Training Models ##########################################


seed = 20031 # Seed is selected as per my Roll No. in the course
train_selection_percent = 0.8 # 80% of the data will be used for training, 20% for testing
predicted = {}

"""
Two evaluation criteria are chosen -

(1) F1 - score: The more the f1-score, the better the classifier
(2) Accuracy: The more the accuracy, the better the classifier
"""

# Scores  will be stored in this dictionary
scores = {}

# Stores confusion matrix for display
confusion_matrix = {}

# Preparing Training  and Testing data
x_train, x_test, y_train, y_test = train_test_split(training_df, class_labels, train_size=train_selection_percent, random_state=seed)


"""
___________________Model_1 - Decision Tree based on the CART algorithm__________________________
"""

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

name = 'Decision Tree' 

predicted[name] = decision_tree.predict(x_test)

scores[name] = metrics.classification_report(y_test, predicted[name])
confusion_matrix[name] = metrics.confusion_matrix(y_test, predicted[name])


"""
___________________Model_2 - Logistic Regression algorithm__________________________
"""

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

name = 'Logistic Regression'

predicted[name] = logreg.predict(x_test)

scores[name] = metrics.classification_report(y_test, predicted[name])
confusion_matrix[name] = metrics.confusion_matrix(y_test, predicted[name])

"""
___________________Model_3 - Random Forest algorithm__________________________
"""

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)

name = 'Random Forest'

predicted[name] =random_forest.predict(x_test)

scores[name] = metrics.classification_report(y_test, predicted[name])
confusion_matrix[name] = metrics.confusion_matrix(y_test, predicted[name])

"""
___________________Model_4 - k-Nearest Neighbours algorithm__________________________
"""

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

name = 'knn'

predicted[name] = knn.predict(x_test)

scores[name] = metrics.classification_report(y_test, predicted[name])
confusion_matrix[name] = metrics.confusion_matrix(y_test, predicted[name])

###################################       Evaluation         #################################


# Display confusion matrix and precision, recall, f1 score and accuracy for all 4 models used above
for item in confusion_matrix:
    print('\n\nConfusion matrix for ' + item)
    display(pd.DataFrame(confusion_matrix[item], index=['Loser', 'Winner'], columns=['Loser', 'Winner']))

for item in scores:
    print(item)
    print(scores[item] + '\n')

print('\n\nRandom Forest Classifier and Decision Tree Classifier have the most F1-score and Accuracy hence they will be chosen for further improvements\n\n')

# Finding the most effective features used for classification, top 10 of them are shown
effective = pd.DataFrame()
effective['feature_name'] = training_df.columns.tolist()
effective['feature_importance'] = random_forest.feature_importances_
effective = effective.sort_values(by='feature_importance', ascending=False)

print('\n\nTop 10 effective features for classification')
display(effective.iloc[0:10:1,:])


###########################  Improvements   ###########################

# Prameters grid for Random Forest Classifier Grid search
rf_grid = { 
    'n_estimators': [300, 400, 500], # No. of trees
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['sqrt', 'log2', 5, 10],
    'min_samples_split': [2,3,4],
    'n_jobs': [-1],
    'class_weight': ['balanced', 'balanced_subsample'],
    'random_state': [seed]
}

# Prameters grid for Decision Tree Classifier Grid search
dt_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [300, 400, 500],
    'min_samples_split': [2,3,4],
    'max_features': ['sqrt', 'log2', 5, 10],
    'random_state': [seed]
}

logreg_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'tol': [1e-3,1e-4,1e-5],
    'C': [0.5,1, 1.5],
    'fit_intercept': [True, False],
    'random_state': [seed],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

knn_grid = {
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'n_jobs': [-1],
    'random_state': [seed]
}

gaussianNB_grid = {
    'var_smoothing': [1e-9, 1e-8,1e-7,1e-6],
    'random_state': [seed]
}

cv_prediction = {}
cv_accuracy = {}
cv_fscore = {}
cv_confusion_matrix = {}
cv_classification_report = {}

# Grid search on Random Forest Classifier
rf_cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_grid, cv=5, scoring='accuracy') #5-fold k-validation based on accuracy score
rf_cv.fit(x_train, y_train)

name = 'Random Forest'

cv_prediction[name] = rf_cv.predict(x_test)
cv_accuracy[name] = round(metrics.accuracy_score(y_test,cv_prediction[name]) * 100, 2)
cv_fscore[name] = round(metrics.f1_score(y_test, cv_prediction[name], average='macro') * 100, 2)
cv_confusion_matrix[name] = pd.DataFrame(metrics.confusion_matrix(y_test, cv_prediction[name]), index=['Loser', 'Winner'], columns=['Loser', 'Winner'])
cv_classification_report[name] = metrics.classification_report(y_test, cv_prediction[name])

# Grid search on Decision Tree Classifier
dt_cv = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=dt_grid, cv=5, scoring='accuracy')
dt_cv.fit(x_train, y_train)

name = 'Decision Tree'

cv_prediction[name] = dt_cv.predict(x_test)
cv_accuracy[name] = round(metrics.accuracy_score(y_test,cv_prediction[name]) * 100, 2)
cv_fscore[name] = round(metrics.f1_score(y_test, cv_prediction[name], average='macro') * 100, 2)
cv_confusion_matrix[name] = pd.DataFrame(metrics.confusion_matrix(y_test, cv_prediction[name]), index=['Loser', 'Winner'], columns=['Loser', 'Winner'])
cv_classification_report[name] = metrics.classification_report(y_test, cv_prediction[name])

# Grid search on Logistic Regression
logreg_cv = GridSearchCV(estimator=LogisticRegression(), param_grid=logreg_grid, cv = 10, scoring='f1_macro')
logreg_cv.fit(x_train,y_train)

name = 'Logistic Regression'

cv_prediction[name] = logreg_cv.predict(x_test)
cv_accuracy[name] = round(metrics.accuracy_score(y_test,cv_prediction[name]) * 100, 2)
cv_fscore[name] = round(metrics.f1_score(y_test, cv_prediction[name], average='macro') * 100, 2)
cv_confusion_matrix[name] = pd.DataFrame(metrics.confusion_matrix(y_test, cv_prediction[name]), index=['Loser', 'Winner'], columns=['Loser', 'Winner'])
cv_classification_report[name] = metrics.classification_report(y_test, cv_prediction[name])

# Grid search on KNN
knn_cv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_grid, cv=10, scoring='f1_macro')
knn_cv.fit(x_train, y_train)

name = 'KNN'

cv_prediction[name] = knn_cv.predict(x_test)
cv_accuracy[name] = round(metrics.accuracy_score(y_test,cv_prediction[name]) * 100, 2)
cv_fscore[name] = round(metrics.f1_score(y_test, cv_prediction[name], average='macro') * 100, 2)
cv_confusion_matrix[name] = pd.DataFrame(metrics.confusion_matrix(y_test, cv_prediction[name]), index=['Loser', 'Winner'], columns=['Loser', 'Winner'])
cv_classification_report[name] = metrics.classification_report(y_test, cv_prediction[name])


for item in cv_accuracy:
    print(item + ' accuracy')
    print(cv_accuracy[item])
    print('')
    print(item + ' F-1 score')
    print(cv_fscore[item])
    print('')
    print(item + ' Classification Report')
    print(cv_classification_report[item])
    print('')
    print(item + ' Confusion Matrix')
    print(cv_confusion_matrix[item])

# Adaptive Boost on Gaussian Naive Bayes
gNB_adaboost = AdaBoostClassifier(estimator=GaussianNB(),n_estimators=300,learning_rate=0.16, random_state=seed)
gNB_adaboost.fit(x_train, y_train)

gNB_adaboost_pred = gNB_adaboost.predict(x_test)

print(metrics.classification_report(y_test, gNB_adaboost_pred))
metrics.f1_score(y_test, gNB_adaboost_pred) * 100
display(metrics.confusion_matrix(y_test, gNB_adaboost_pred))

###########################          Result             ##########################

# Importing the test data
test = pd.read_csv(parent_dir + '/data/test.csv')

# Creating dataframe from test data to put into the classifier
temp_.clear()
list_first = test.iloc[:, 0].to_list()

for i in range(len(list_first)):
    temp_.append(get_features(list_first[i], first_pokemon).to_frame().T)

df_first_pokemon_test = pd.concat(temp_, ignore_index=True)

temp_.clear()
list_second = test.iloc[:, 1].to_list()

for i in range(len(list_second)):
    temp_.append(get_features(list_second[i], second_pokemon).to_frame().T)

df_second_pokemon_test = pd.concat(temp_, ignore_index=True)

df_first_pokemon_test = df_first_pokemon_test.drop([df_first_pokemon_test.columns[0], df_first_pokemon_test.columns[1]], axis=1)
df_second_pokemon_test = df_second_pokemon_test.drop([df_second_pokemon_test.columns[0],df_second_pokemon_test.columns[1]], axis=1)

# Creating final test dataframe to put into the classifier
test_df = pd.concat([df_first_pokemon_test, df_second_pokemon_test], axis=1)

# Since Random Forest Classifier had the best f-1 score and accuracy, it is chosen to run on the test data
result_rf = rf_cv.predict(test_df)

result_id_rf = []

for i in range(len(result_rf)):
    if (result_rf[i] == 1):
        result_id_rf.append(list_first[i])
    else:
        result_id_rf.append(list_second[i])

# Creating the predicted data in the same format as in 'combats.csv' and exporting to a file 'results.csv'
output = pd.DataFrame({'First_pokemon': list_first , 'Second_pokemon': list_second, 'Winner': result_id_rf})
output.to_csv(parent_dir + '/data/result.csv', index=False)
print('\n\nResult')
display(output)