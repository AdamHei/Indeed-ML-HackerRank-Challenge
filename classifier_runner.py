import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score


# Returns the accuracy over a given number of iterations with a given classifier
def runner(data, classes, classifier, num_iterations=20):
    scores = np.zeros(num_iterations)

    for i in range(0, num_iterations):
        score = cross_val_score(classifier, data, classes, cv=10)
        scores[i] = np.array(score).mean()

    return scores.mean()

    # for i in range(0, num_iterations):
    #     x_train, x_test, y_train, y_test = train_test_split(data, classes, .7, random_state=42)
    #
    #     classifier.fit(x_train, y_train)
    #     prediction = classifier.predict(x_test)
    #     scores.append(cross_val_score(y_test, prediction))
