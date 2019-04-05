from GES import GES
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from metrics import f_score, bac
from sklearn.base import clone
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold
import helper as h
import numpy as np
from tqdm import tqdm

metrics = {
    "b": bac,
    "g": geometric_mean_score,
    "f": f_score,
}
classifiers = {
    'LR': LogisticRegression(solver="lbfgs"),
    'RS-LR': GES(base_clf=LogisticRegression(solver="lbfgs"),num_iter=0),
    'GES-LR': GES(base_clf=LogisticRegression(solver="lbfgs")),
    'GES-RA-LR': GES(base_clf=LogisticRegression(solver="lbfgs"), alpha=.1),
    'GES-RB-LR': GES(base_clf=LogisticRegression(solver="lbfgs"), alpha=.1, beta=.1),
    'NB': GaussianNB(),
    'RS-NB': GES(base_clf=GaussianNB(),num_iter=0),
    'GES-NB': GES(base_clf=GaussianNB()),
    'GES-RA-NB': GES(base_clf=GaussianNB(), alpha=.1),
    'GES-RB-NB': GES(base_clf=GaussianNB(), alpha=.1, beta=.1),
}
folds = 5

datasets = h.datasets()

for dataset in datasets:
    if dataset not in ["hepatitis"]:
        continue
    print(dataset)
    X, y, _, _ = h.load_dataset(dataset)
    label_corrector = np.max(y) - 1

    scores = np.zeros((folds, len(classifiers), len(metrics)))

    skf = StratifiedKFold(n_splits=folds, random_state=1337)
    for f, (train, test) in tqdm(enumerate(skf.split(X, y)), ascii=True, total=folds):
        X_train, X_test = X[train], X[test]
        y_train, y_test = (y[train] - label_corrector, y[test] - label_corrector)

        for m, metric in tqdm(enumerate(metrics), ascii=True, total=len(metrics)):
            for c, cid in tqdm(enumerate(classifiers),ascii=True,total=len(classifiers)):
                clf = clone(classifiers[cid])
                if cid.find('GES') == 0:
                    clf.metric=metrics[metric]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                score = metrics[metric](y_test, y_pred)

                scores[f, c, m] = score


    np.save("results/%s" % dataset, scores)
    averaged = np.mean(np.mean(scores, axis=0), axis=1)
    print("\n")
    for i, c in enumerate(classifiers):
        print("%9s - %.3f" % (c, averaged[i]))
