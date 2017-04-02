from sklearn import svm
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from classifier_runner import runner
from parser import get_data


def driver():
    data, classes = get_data()
    classifiers = [QuadraticDiscriminantAnalysis(), LinearDiscriminantAnalysis(),
                   KNeighborsClassifier(), GaussianNB(), svm.SVC(), tree.DecisionTreeClassifier()]

    for tag in classes:
        for classifier in [tree.DecisionTreeClassifier()]:
            score = runner(data, classes[tag], classifier, num_iterations=1)
            # print(classifier, score)
            print("Accuracy for tag {0} was {1}%".format(tag, score * 100))
            print()


if __name__ == '__main__':
    driver()
