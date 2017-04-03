import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

classes = {'part-time-job': np.zeros(4375), 'full-time-job': np.zeros(4375), 'hourly-wage': np.zeros(4375),
           'salary': np.zeros(4375), 'associate-needed': np.zeros(4375), 'bs-degree-needed': np.zeros(4375),
           'ms-or-phd-needed': np.zeros(4375), 'licence-needed': np.zeros(4375),
           '1-year-experience-needed': np.zeros(4375), '2-4-years-experience-needed': np.zeros(4375),
           '5-plus-years-experience-needed': np.zeros(4375), 'supervising-job': np.zeros(4375)}


def get_data():
    data = np.genfromtxt("train.tsv", delimiter="\t", dtype=None)
    data = np.char.decode(data)
    data = data[1:, :]

    tags = data[:, 0]
    descriptions = data[:, 1]

    vectorizer = TfidfVectorizer(min_df=1)

    fitted = vectorizer.fit_transform(descriptions)

    for i in range(0, tags.shape[0]):
        tag_arr = tags[i].split(" ")
        for tag in tag_arr:
            if tag != '':
                classes[tag][i] = 1

    return fitted, classes

# scores = []
# classifier = QuadraticDiscriminantAnalysis()
#
# for i in range(0, 20):
#     x_train, x_test, y_train, y_test = train_test_split(fitted.toarray(), classes['part-time-job'], test_size=.7,
#                                                         random_state=42)
#     classifier.fit(x_train, y_train)
#     prediction = classifier.predict(x_test)
#     scores.append(accuracy_score(y_test, prediction))
#
# avg_score = np.array(scores).mean()
# print("Success rate: {0}%".format(avg_score * 100))
