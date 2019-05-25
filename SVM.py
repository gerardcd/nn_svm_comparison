import math
from sklearn import svm
from sklearn.model_selection import GridSearchCV

N = 1024

parameters = {'kernel':['rbf', 'linear'], 'C':[0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100]}
cv = int(math.log(N) / math.log(2))

clf = svm.SVC()
clf = GridSearchCV(clf, parameters, cv=cv, verbose=1)

def encode(value):
    value = int(value)

    return 1 if value == 1 else 2

def dataSetFromFile(fileName):
    xData = []
    yData = []
    with open(fileName) as file:
        for row in file:
            values = row.split()

            xData.append(values[:-1])

            yData.append(
                encode(values[-1])
            )

    return (xData, yData)


(xTrain, yTrain) = dataSetFromFile('shuttle.trn')
(xTest, yTest) = dataSetFromFile('shuttle.tst')

xTrain = xTrain[:N]
yTrain = yTrain[:N]

clf.fit(xTrain, yTrain)

pred = clf.predict(xTest)

errors = 0

for i in range(len(pred)):
    if pred[i] != yTest[i]:
        errors += 1

print("Best Kernel: ", clf.best_estimator_.kernel)
print("Best C: ", clf.best_estimator_.C)
print("Best Gamma: ", clf.best_estimator_.gamma)

print("Test accuracy: ", (len(pred) - errors) / len(pred))