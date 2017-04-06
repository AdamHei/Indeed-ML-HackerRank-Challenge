import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from main.classifier_runner import runner
from main.knn_runner import knn_with_cross_fold_validation
from main.myparser import get_data, get_test_data


def just_trees_printer():
    _, classes = get_data()
    data, test_data = get_test_data()

    # data = SelectKBest(chi2, k=500).fit_transform(data, classes['hourly-wage'])

    # tree_classifier = RandomForestClassifier(n_estimators=10)
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    pred_arr = np.empty([test_data.shape[0], 12], dtype='S50')

    i = 0
    for tag in classes:
        print("Now on tag", tag)
        knn_classifier.fit(data, classes[tag])

        pred = knn_classifier.predict(test_data)

        for index in range(0, pred.shape[0]):
            if pred[index] == 1.0:
                temp = tag
            else:
                temp = ''
            pred_arr[index][i] = temp
        i += 1

    decoded = np.char.decode(pred_arr)
    outfile = open("tags.tsv", 'w')
    outfile.write('tags\n')

    for row in decoded:
        for elem in row:
            if elem != '':
                outfile.write('{0}\t'.format(elem))
        outfile.write('\n')

    outfile.close()

    # np.savetxt("predictions.txt", np.char.decode(pred_arr), delimiter='\t', fmt='%s')


def driver():
    data, classes = get_data()

    data = SelectKBest(chi2, k=500).fit_transform(data, classes['hourly-wage'])

    data_dense = data.toarray()

    picky_classifiers = ["QDA", "LDA", "Naive Bayes"]

    classifier_map = {"SVM": svm.SVC(), "RandomForest": RandomForestClassifier(n_estimators=10),
                      "QDA": QuadraticDiscriminantAnalysis(), "LDA": LinearDiscriminantAnalysis(),
                      "Naive Bayes": GaussianNB()}

    for tag in classes:
        print("Starting on tag {0}".format(tag))

        num_iters = 1

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

        print(
            "\nBest classifier for tag {0} was {1} with accuracy {2}%\n".format(tag, best_classifier, max_score * 100))


just_trees_printer()
