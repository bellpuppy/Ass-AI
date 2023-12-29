'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

numFold = 10

def k_fold_split(X, y, num_fold):
    n = len(X)
    fold_size = n // num_fold
    remainder = n % num_fold

    indices = list(range(n))

    folds = []

    for fold in range(num_fold):
        # Tính kích thước của fold
        fold_end = fold * fold_size + min(fold, remainder) + fold_size

        # Chia dữ liệu thành fold kiểm tra và dữ liệu huấn luyện
        test_indices = indices[fold * fold_size:min(fold_end, n)]
        train_indices = indices[:fold * fold_size] + indices[fold_end:n]

        X_train = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]

        X_test = [X[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]


    return X_train, y_train, X_test, y_test

    

def evaluatePerformance(numTrials=100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = r"C:\python\dtree\dtree_learning\Assignment1\hw1_skeleton\data\SPECTF.dat"
    data = np.loadtxt(filename, delimiter=',') # Data: mảng 267 hàng 45 cột 
    X = data[:, 1:] # Tất cả hàng cột trừ cột đầu tiên => mảng 267 hàng 44 cột
    y = np.array([data[:, 0]]).T # Cột đầu tiên transpose => mảng 1 cột chứa label
    n,d = X.shape

    # create list to hold data
    treeAccuracies = []
    stumpAccuracies = []
    dt3Accuracies = []

    # perform 100 trials
    for x in range(0, numTrials):
        print(x)
        # shuffle the data
        idx = np.arange(n) # Khoi tao mang idx so nguyen co gia tri tu 1 toi idx
        np.random.seed(13) # Khoi tao mang ngau nhien
        np.random.shuffle(idx) # Trộn lại mảng idx, các phần tử nằm vị trí ngẫu nhiên
        X = X[idx] # Phương thức ngẫu nhiên hóa các phần tử của mảng X dựa vào vị trí chỉ số mảng idx cung cấp
        y = y[idx] # Tương tự trên

        Xtrain,ytrain,Xtest,ytest = k_fold_split(X,y, numFold)
        
        # Train decision tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(Xtrain,ytrain)
        # output predictions on the remaining data
        y_pred_tree = clf.predict(Xtest)

        # train the 1-level decision tree
        oneLevel = tree.DecisionTreeClassifier(max_depth=1)
        oneLevel = oneLevel.fit(Xtrain,ytrain)
        # output predictions on the remaining data
        y_pred_stump = oneLevel.predict(Xtest)

        # train the 3-level decision tree
        threeLevel = tree.DecisionTreeClassifier(max_depth=3)
        threeLevel = threeLevel.fit(Xtrain,ytrain)
        # output predictions on the remaining data
        y_pred_dt3 = threeLevel.predict(Xtest)

        # Tính toán training accuracy của model và save vào list chứa tất cả độ chính xác
        treeAccuracies.append(accuracy_score(ytest, y_pred_tree))
        stumpAccuracies.append(accuracy_score(ytest, y_pred_stump))
        dt3Accuracies.append(accuracy_score(ytest, y_pred_dt3))

        percentage_values = range(10, 101, 10)
        mean_accuracy = []
        std_accuracy = []
        treeacc_new = []
        
        for percentage in percentage_values:
            # Calculate the number of samples to use for the given percentage
            num_samples = int(len(Xtrain) * percentage / 100)
            
            Xtrain_new = Xtrain[:num_samples ]
            ytrain_new = ytrain[:num_samples]
            Xtest_new = Xtest[:num_samples]
            ytest_new = ytest[:num_samples]
            # Train decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(Xtrain_new,ytrain_new)
            # output predictions on the remaining data
            y_pred_new = clf.predict(Xtest_new)
            treeacc_new.append(accuracy_score(ytest_new, y_pred_new))
            # Calculate mean and std of test accuracy
            mean_accuracy.append(np.mean(treeacc_new))
            std_accuracy.append(np.std(treeacc_new))
    plt.errorbar(percentage_values, mean_accuracy, yerr=std_accuracy, fmt='-o')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Test Accuracy')
    plt.title('Learning Curve with K-Fold Cross-Validation')
    plt.show()
    # Tính toán mean và std của các accuracy thu được
    meanDecisionTreeAccuracy = np.mean(treeAccuracies)
    stddevDecisionTreeAccuracy = np.std(treeAccuracies)
    meanDecisionStumpAccuracy = np.mean(stumpAccuracies)
    stddevDecisionStumpAccuracy = np.std(stumpAccuracies)
    meanDT3Accuracy = np.mean(dt3Accuracies)
    stddevDT3Accuracy = np.std(dt3Accuracies)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    


    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print ("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print ("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print ("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")

# ...to HERE.