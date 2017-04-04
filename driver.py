from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from classifier_runner import runner
from knn_runner import knn_with_cross_fold_validation
from parser import get_data


def driver():
    data, classes = get_data()
    data_dense = data.toarray()

    picky_classifiers = ["QDA", "LDA", "Naive Bayes"]

    classifier_map = {"SVM": svm.SVC(), "RandomForest": RandomForestClassifier(n_estimators=10),
                      "QDA": QuadraticDiscriminantAnalysis(), "LDA": LinearDiscriminantAnalysis(),
                      "Naive Bayes": GaussianNB()}

    for tag in classes:
        print("Starting on tag {0}".format(tag))

        num_iters = 20

        best_k = knn_with_cross_fold_validation(data, classes[tag], num_iterations=num_iters)

        best_classifier = None
        max_score = 0.0

        score = runner(data, classes[tag], KNeighborsClassifier(n_neighbors=best_k), num_iterations=num_iters)

        if score > max_score:
            max_score = score
            best_classifier = "KNN"

        for classifier in classifier_map:
            score = runner(data_dense if classifier in picky_classifiers else data,
                           classes[tag], classifier_map[classifier], num_iterations=num_iters)

            if score > max_score:
                max_score = score
                best_classifier = classifier

                # print("Accuracy for tag {0} with {1} iterations was {2}%".format(tag, num_iters, score * 100))

        print("\nBest classifier for tag {0} was {1} with accuracy {2}%\n".format(tag, best_classifier, max_score * 100))


if __name__ == '__main__':
    driver()
