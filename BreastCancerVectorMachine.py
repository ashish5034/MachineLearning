# Breast cancer dataset with support vector machine
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def MySVM():
    # load dataset
    cancer = datasets.load_breast_cancer()
    
    # print the name of the 13 features 
    print("Features of the cancer datset : ",cancer.feature_names)
    
    # print the label type of cancer('malignant''benign)
    print("Labels of the cancer dataset : ",cancer.target_names)
    
    # print data(feature)shape
    print("Shape of dataset is : ",cancer.data.shape)
    
    # print the cancer data features (top 5 records)
    print("First 5 records are : ")
    print(cancer.data[0:5])
    
    # print the cancer label (0:malignant, 1:benign)
    print("Target of dataset : ",cancer.target)
    
    # split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109) #70% traing and 30% test
    
    # create a SVM classifier
    clf = svm.SVC(kernel='linear') #linear kernel
    
    # Train the model using the training sets
    clf.fit(X_train,y_train)
    
    # predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    # print predicted data 
    print("Predicted data : ",y_pred)
    
    # Model Accuracy: how often is the classifier coreect?
    print("Accuracy of the model is : ",metrics.accuracy_score(y_test,y_pred)*100)
    
def main():
    print("____Support Vector Machine____")
    MySVM()
if __name__ == "__main__":
    main()