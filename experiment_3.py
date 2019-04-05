from GES import GES
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from metrics import f_score, bac
from sklearn.base import clone
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold

n_features_ = [10,50,100,500]
n_features_ = [1000]

metrics = {
    "b": bac,
    "g": geometric_mean_score,
    "f": f_score,
}
classifiers = {
    'LR': LogisticRegression(solver="lbfgs"),
    'RS_LR': GES(base_clf=LogisticRegression(solver="lbfgs"),num_iter=0),
    'GES_LR': GES(base_clf=LogisticRegression(solver="lbfgs"),num_iter=50),
    'GES_NB': GES(base_clf=GaussianNB(),num_iter=200),
    'RS_NB': GES(base_clf=GaussianNB(),num_iter=0),
    'NB': GaussianNB(),
}
folds = 5

for n_features in n_features_:
    print(n_features)
    X, y = make_classification(n_samples=1000, n_features=50,
                               n_informative=10,
                               n_clusters_per_class=1,
                               #n_redundant=10,
                               #n_repeated=0,
                               weights=(.5,.5), random_state=n_features,
                               shuffle=True, flip_y=.1)

    for metric in metrics:
        print(metric)
        skf = StratifiedKFold(n_splits=folds, random_state=1337)
        for f, (train, test) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            for cid in classifiers:
                clf = clone(classifiers[cid])
                if cid.find('GES') == 0:
                    clf.metric=metrics[metric]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                score = metrics[metric](y_test, y_pred)
                print("%6s - %.3f" % (cid, score))
            break
