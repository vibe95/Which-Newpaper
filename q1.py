'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression



def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    test_pred = model.predict(binary_test)
    
    accuracy_train = (train_pred == train_labels).mean()
    accuracy_test = (test_pred == test_labels).mean()
    
    print('BernoulliNB train accuracy = {}'.format(accuracy_train))
    print('BernoulliNB test accuracy = {}'.format(accuracy_test))
    print('BernoulliNB train zero-one loss = {}'.format(1 - accuracy_train))        
    print('BernoulliNB test zero-one loss = {}'.format(1 - accuracy_test))        

    return model



def random_forest (bow_train, train_labels, bow_test, test_labels):
    # training the Random Forest model

    #Use grid search to get the best parameters
    parameters = {'max_depth':[1, 30], 'n_estimators':[1,40]}
    rndFr = RandomForestClassifier()
    clf = GridSearchCV(rndFr, parameters)
    clf.fit(bow_train, train_labels) 

    #evaluate the Random Forest model
    train_pred = clf.predict(bow_train)
    test_pred = clf.predict(bow_test)
    
    accuracy_train = (train_pred == train_labels).mean()
    accuracy_test = (test_pred == test_labels).mean()
    
    print('Random Forest train accuracy = {}'.format(accuracy_train))
    print('Random Forest test accuracy = {}'.format(accuracy_test))
    print('Random Forest train zero-one loss = {}'.format(1 - accuracy_train))        
    print('Random Forest test zero-one loss = {}'.format(1 - accuracy_test))    

    return clf


def multi_bern(bow_train, train_labels, bow_test, test_labels):
    # training the MultinomialNB model

    clf = MultinomialNB()
    clf.fit(bow_train, train_labels) 

    #evaluate the MultinomialNB model
    train_pred = clf.predict(bow_train)
    test_pred = clf.predict(bow_test)
    
    accuracy_train = (train_pred == train_labels).mean()
    accuracy_test = (test_pred == test_labels).mean()
    
    print('MultinomialNB train accuracy = {}'.format(accuracy_train))
    print('MultinomialNB test accuracy = {}'.format(accuracy_test))
    print('MultinomialNB train zero-one loss = {}'.format(1 - accuracy_train))        
    print('MultinomialNB test zero-one loss = {}'.format(1 - accuracy_test))    

    return clf

def logistic(bow_train, train_labels, bow_test, test_labels):
    # training the logisitic model

    clf = LogisticRegression()
    clf.fit(bow_train, train_labels) 

    #evaluate the logisitic model
    train_pred = clf.predict(bow_train)
    test_pred = clf.predict(bow_test)
    
    accuracy_train = (train_pred == train_labels).mean()
    accuracy_test = (test_pred == test_labels).mean()
    
    print('Logistic train accuracy = {}'.format(accuracy_train))
    print('Logistic test accuracy = {}'.format(accuracy_test))
    print('Logistic train zero-one loss = {}'.format(1 - accuracy_train))        
    print('Logistic test zero-one loss = {}'.format(1 - accuracy_test))    

    return clf

def confusion(model, bow_test, test_labels):
    #Find the confusion matrix for the given model
    predictions = model.predict(bow_test)
    
    conf_matrix = np.zeros((20,20),dtype=int) # create a square matrix for number of class 20
    
    #For each prediction increase count in relavent cell based on predicition vs true value
    for i in range(0,len(predictions)):
        pred_class = predictions[i]
        true_class = test_labels[i]
        
        #if the predicted value == true value then the the identiy column of the matrix will be updated with the count
        #else the true row by predicted column will be updated
        #print(pred_class,", ",true_class)
        conf_matrix[true_class,pred_class] = conf_matrix[true_class,pred_class] + 1
            
            
    #print and return matrix
    print(conf_matrix)          
    return conf_matrix


## OTHER METHODS I TRIED
#def knn_line(bow_train, train_labels, bow_test, test_labels):
    ## training the baseline model

    #neigh = KNeighborsClassifier(n_neighbors=10)

    #neigh.fit(bow_train, train_labels) 

    ##evaluate the baseline model
    #train_pred = neigh.predict(bow_train)
    #print('KNN train accuracy = {}'.format((train_pred == train_labels).mean()))
    #test_pred = neigh.predict(bow_test)
    #print('KNN test accuracy = {}'.format((test_pred == test_labels).mean()))

    #return neigh
    
#def neural_net(bow_train, train_labels, bow_test, test_labels):
    ## training the baseline model

    #model = MLPClassifier()

    #model.fit(bow_train, train_labels) 

    ##evaluate the baseline model
    #train_pred = model.predict(bow_train)
    #print('Neural net train accuracy = {}'.format((train_pred == train_labels).mean()))
    #test_pred = model.predict(bow_test)
    #print('Neural net test accuracy = {}'.format((test_pred == test_labels).mean()))


    ##evaluate the baseline model
    #train_pred = model.predict(bow_train)
    #test_pred = model.predict(bow_test)
    
    #accuracy_train = (train_pred == train_labels).mean()
    #accuracy_test = (test_pred == test_labels).mean()
    
    #print('Neural Net train accuracy = {}'.format(accuracy_train))
    #print('Neural Net test accuracy = {}'.format(accuracy_test))
    #print('Neural Net train zero-one loss = {}'.format(1 - accuracy_train))        
    #print('Neural Net test zero-one loss = {}'.format(1 - accuracy_test))    
    
    #return model

    
#def svm_line(bow_train, train_labels, bow_test, test_labels):
    ## training the baseline model

    #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    #svc = svm.SVC()
    #clf = GridSearchCV(svc, parameters)
    #clf.fit(bow_train, train_labels) 

    ##evaluate the baseline model
    #train_pred = clf.predict(bow_train)
    #print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    #test_pred = clf.predict(bow_test)
    #print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    #return clf
if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = tf_idf_features(train_data, test_data)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    
    ##Knn did not get good accuracy and svm/nn take too long
    #KNN_model = knn_line(train_bow, train_data.target, test_bow, test_data.target)
    #svm_model = svm_line(train_bow, train_data.target, test_bow, test_data.target)
    #NN_model = neural_net(train_bow, train_data.target, test_bow, test_data.target)
    
    #What Works
    multi_bern_model = multi_bern(train_bow, train_data.target, test_bow, test_data.target)
    random_forest_model = random_forest(train_bow, train_data.target, test_bow, test_data.target)
    logistic_model = logistic(train_bow, train_data.target, test_bow, test_data.target)

    #find the confussion model for the best classiefier: Logistic  
    confusion(logistic_model, test_bow, test_data.target) 
    