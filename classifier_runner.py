import numpy as np
from sklearn.model_selection import cross_val_score


# Returns the accuracy over a given number of iterations with a given classifier
def runner(data, classes, classifier, num_iterations=20):
    scores = np.zeros(num_iterations)

    for i in range(0, num_iterations):
        print("Now on iteration {0}".format(i))

        score = cross_val_score(classifier, data, classes, cv=10)
        scores[i] = np.array(score).mean()

    return scores.mean()
