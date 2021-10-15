import numpy as np
import math
import matplotlib.pyplot as plt

def NB_GenerateData(D, p, count):
    '''
    This function generates data to follow the probabilities that the Naive Bayes model learned
    D is the matrix of P(X|Y) probabilities
    p is the P(Y) prior probabilities
    count is the number of data points we will generate
    '''
    xData = np.zeros((count, D.shape[1]))
    countTrue = np.sum(np.random.binomial(1, p, count))
    countFalse = count-countTrue
    trueY = np.ones(countTrue)+1
    falseY = np.ones(countFalse)
    yData = np.concatenate((trueY, falseY))
    for f in range(D.shape[1]):
        trueX = np.random.binomial(1,D[1][f],countTrue)
        falseX = np.random.binomial(1,D[0][f],countFalse)
        tempX = np.concatenate((trueX, falseX))
        xData[:,f] = tempX
    return xData, yData

def augmentFeatures(XTrain, yTrain, XTest, yTest):
    te_vals = {}
    teaug_vals = {}

    np.random.seed(0)
    for i in range(150,451,30):
        te_vals[i] = 0
        teaug_vals[i] = 0

    for i in range(10):
        fset = np.random.choice(XTrain.shape[1],50,replace=False)

        for m in range(150,451,30):
            D = NB_XGivenY(XTrain[0:m], yTrain[0:m])
            p = NB_YPrior(yTrain[0:m])

            yHatTest = NB_Classify(D, p, XTest)

            testAcc = NB_ClassificationAccuracy(yHatTest, yTest)

            te_vals[m] += testAcc
            
            #augmebting features
            xtrain_aug = np.repeat(XTrain[0:m,fset],100,axis=1)
            xtest_aug = np.repeat(XTest[:,fset],100,axis=1)
            xTrain = np.hstack([XTrain[0:m],xtrain_aug])
            xTest = np.hstack([XTest,xtest_aug])
            D = NB_XGivenY(xTrain, yTrain[0:m])
            p = NB_YPrior(yTrain[0:m])

            yHatTest = NB_Classify(D, p, xTest)

            testAcc = NB_ClassificationAccuracy(yHatTest, yTest)
            teaug_vals[m] += testAcc

            

    for i in range(150,451,30):
        te_vals[i] /= 10
        teaug_vals[i] /= 10

    plt.plot(list(te_vals.keys()),list(te_vals.values()),label='Original dataset')
    plt.plot(list(teaug_vals.keys()),list(teaug_vals.values()),label='With augmented features')
    plt.legend()
    plt.xlabel('m')
    plt.ylabel('Test Accuracy')
    plt.savefig('./q9.pdf')


def NB_XGivenY(XTrain, yTrain, a=0.001, b=0.9):

    THRESHOLD = 1e-5
    
    # TODO: Implement P(X|Y) with Beta(a,1+b) Prior
    D = np.zeros((2, XTrain.shape[1]))
    class1_index = np.where(yTrain == 1)[0]
    class2_index = np.where(yTrain == 2)[0]
    for i in range(XTrain.shape[1]):
        wi_c1 = XTrain[:, i][class1_index]
        D[0, i] = (a + np.count_nonzero(wi_c1)) / (a + b + wi_c1.shape[0])
        wi_c2 = XTrain[:, i][class2_index]
        D[1, i] = (a + np.count_nonzero(wi_c2)) / (a + b + wi_c2.shape[0])
    
    D = np.clip(D, THRESHOLD, 1-THRESHOLD)
    return D


def NB_YPrior(yTrain):
    # TODO: Implement P(Y)
    return np.where(yTrain == 1)[0].shape[0] / yTrain.shape[0]


def NB_Classify(D, p, X):
    # TODO: Implement Label Predition Vector of X
    cond_prob_c1 = np.zeros(X.shape)
    D_c1 = np.repeat(D[0, :].reshape((1, -1)), X.shape[0], axis=0)
    cond_prob_c1[np.where(X == 1)] = D_c1[np.where(X == 1)]
    cond_prob_c1[np.where(X == 0)] = 1 - D_c1[np.where(X == 0)]
    c1 = np.sum(np.log(cond_prob_c1), axis=1).reshape((-1, 1))
    c1 = c1 + np.log(p)

    cond_prob_c2 = np.zeros(X.shape)
    D_c2 = np.repeat(D[1, :].reshape((1, -1)), X.shape[0], axis=0)
    cond_prob_c2[np.where(X == 1)] = D_c2[np.where(X == 1)]
    cond_prob_c2[np.where(X == 0)] = 1 - D_c2[np.where(X == 0)]
    c2 = np.sum(np.log(cond_prob_c2), axis=1).reshape((-1, 1))
    c2 = c2 + np.log(1-p)

    yHat = np.zeros(c1.shape)
    yHat[c1 > c2] = 1
    yHat[c1 <= c2] = 2

    return yHat


def NB_ClassificationAccuracy(yHat, yTruth):

    # TODO: Compute the Classificaion Accuracy of yHat Against yTruth
    return np.where(yHat == yTruth)[0].shape[0] / yTruth.shape[0]


if __name__ == "__main__":

    import pickle
    with open("hw1data.pkl", "rb") as f:
        data = pickle.load(f)

    XTrain = data["XTrain"]
    yTrain = data["yTrain"]

    XTest = data["XTest"]
    yTest = data["yTest"]

    Vocab = data["Vocabulary"]
    
    # Q5.5
    D = NB_XGivenY(XTrain, yTrain)
    p = NB_YPrior(yTrain)
    yHatTrain = NB_Classify(D, p, XTrain)
    yHatTest = NB_Classify(D, p, XTest)
    trainAcc = NB_ClassificationAccuracy(yHatTrain, yTrain)
    testAcc = NB_ClassificationAccuracy(yHatTest, yTest)
    print("Train Acc:", trainAcc)
    print("Test Acc:", testAcc)

    # Q5.6
    trainAccs = []
    testAccs = []
    m_range = list(range(100, 450, 30))
    m_range.append(450)

    for m in m_range:
        D = NB_XGivenY(XTrain[:m], yTrain[:m])
        p = NB_YPrior(yTrain[:m])
        yHatTrain = NB_Classify(D, p, XTrain)
        yHatTest = NB_Classify(D, p, XTest)
        trainAccs.append(NB_ClassificationAccuracy(yHatTrain, yTrain))
        testAccs.append(NB_ClassificationAccuracy(yHatTest, yTest))
    
    plt.plot(m_range, trainAccs, label='Train Acc')
    plt.plot(m_range, testAccs, label='Test Acc')
    plt.xlabel('m')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Q5.7
    D = NB_XGivenY(XTrain, yTrain)
    
    print("First Metric, Y=1")
    top_5_index_1 = np.flip(np.argsort(D[0, :])[-5:])
    for index in top_5_index_1:
        print(Vocab[index, 0])
    
    print("First Metric, Y=2")
    top_5_index_2 = np.flip(np.argsort(D[1, :])[-5:])
    for index in top_5_index_2:
        print(Vocab[index, 0])

    print("Second Metric, Y=1")
    top_5_index_3 = np.flip(np.argsort(D[0, :] /  D[1, :])[-5:])
    for index in top_5_index_3:
        print(Vocab[index, 0])

    print("Second Metric, Y=2")
    top_5_index_4 = np.flip(np.argsort(D[1, :] /  D[0, :])[-5:])
    for index in top_5_index_4:
        print(Vocab[index, 0])

    # Q5.8
    D = NB_XGivenY(XTrain, yTrain)
    p = NB_YPrior(yTrain)

    importance1 = np.abs(1 - (D[0, :] / D[1, :]))
    importance2 = np.abs(1 - (D[1, :] / D[0, :]))

    for i, importance in enumerate([importance1, importance2]):
        train_accs, test_accs = [], []
        for t in np.arange(0.1, 1.05, 0.1):
            D_indexed = D[np.ix_(np.arange(0, 2), np.where(importance > t)[0])]
            XTrain_indexed = XTrain[np.ix_(np.arange(0, XTrain.shape[0]), np.where(importance > t)[0])]
            XTest_indexed = XTest[np.ix_(np.arange(0, XTest.shape[0]), np.where(importance > t)[0])]
            yHat_train = NB_Classify(D_indexed, p, XTrain_indexed)
            train_acc = NB_ClassificationAccuracy(yHat_train, yTrain)
            yHat_test = NB_Classify(D_indexed, p, XTest_indexed)
            test_acc = NB_ClassificationAccuracy(yHat_test, yTest)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print("T:", t)
            print(train_acc, test_acc)
            print("remove words count:", np.where(importance < t)[0].shape[0])

        plt.plot(np.arange(0.1, 1.05, 0.1), train_accs, label='Train Acc')
        plt.plot(np.arange(0.1, 1.05, 0.1), test_accs, label='Test Acc')
        plt.xlabel('T')
        plt.ylabel('Accuracy')
        plt.title("I" + str(i+1))
        plt.legend()
        plt.show()
        plt.clf()

    # Q5.9
    augmentFeatures(XTrain, yTrain, XTest, yTest)

    # Q5.10
    xData, yData = NB_GenerateData(D, p, 1000)
    yHat_new = NB_Classify(D, p, xData)
    acc = NB_ClassificationAccuracy(yHat_new, yData.reshape((-1, 1)))
    print(acc)
