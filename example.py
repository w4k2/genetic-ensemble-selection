from GES import GES
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from metrics import f_score, recall, precision, bac
import time
import cProfile
from imblearn.metrics import geometric_mean_score

m = [f_score, geometric_mean_score, bac, recall, precision]

# Prepare dataset
X, y = make_classification(
    n_samples=200, n_features=10, weights=(0.8, 0.2), random_state=13
)

# Divide it
for metric in m:
    skf = StratifiedKFold(n_splits=5, random_state=13, shuffle=True)
    for fold, (train, test) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        # Build model LR
        start = time.time()
        clf = GES(
            metric=metric,
            base_clf=LogisticRegression(solver="lbfgs"),
            num_iter=100,
            beta=0.1,
            alpha=0.1,
        )
        clf.fit(X_train, y_train)
        # exit()
        y_pred = clf.predict(X_test)
        score = metric(y_test, y_pred)
        print("GES LR = %.3f %.3f" % (score, time.time() - start))

        # Build model GNB
        clf = GES(metric=metric, base_clf=GaussianNB())
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        score = metric(y_test, y_pred)
        print("GES NB = %.3f %.3f" % (score, time.time() - start))

        # exit()
