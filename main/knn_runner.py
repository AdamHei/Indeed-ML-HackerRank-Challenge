import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def knn_with_cross_fold_validation(data, classes, num_iterations=20):
    possible_ks = np.arange(1, 100, step=2)
    cv_scores = np.empty([50, num_iterations])

    for i in range(0, num_iterations):
        for k in possible_ks:
            if k % 11 == 0:
                print("k={0}".format(k))
            knn = KNeighborsClassifier(n_neighbors=k)
            # Took out scoring='accuracy'
            scores = cross_val_score(knn, data, classes, cv=10)
            cv_scores[int(k / 2)][i] = (scores.mean())

    mean_cv_errors = [arr.mean() for arr in cv_scores]
    maximum = max(mean_cv_errors)
    index = mean_cv_errors.index(maximum)

    optimal_k = possible_ks[index]
    print('The best k with {0} iterations was {1} with a success rate of {2}%'
          .format(num_iterations, optimal_k, max(mean_cv_errors) * 100))

    # inverse_errors = [1 - x for x in mean_cv_errors]
    # plotter(possible_ks,
    #         inverse_errors,
    #         'Number of Neighbors K',
    #         'Misclassification Rate',
    #         'KNN with Cross Fold Validation (Best k={0})'.format(possible_ks[index]),
    #         num_iterations)

    return optimal_k


def plotter(x_set, y_set, x_lab, y_lab, title, iterations):
    plt.plot(x_set, y_set)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title("{0} with {1} iterations".format(title, iterations))

    plt.savefig(title)
    # plt.show()
