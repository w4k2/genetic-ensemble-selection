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
from scipy.stats import kruskal, wilcoxon, ttest_ind

p_t = .05
test = ttest_ind
metrics = {
    "b": bac,
    "g": geometric_mean_score,
    "f": f_score,
}
classifiers = {
    'LR': LogisticRegression(solver="lbfgs"),
    'RS-LR': GES(base_clf=LogisticRegression(solver="lbfgs"), num_iter=0),
    'GES-LR': GES(base_clf=LogisticRegression(solver="lbfgs")),
    'GES-RA-LR': GES(base_clf=LogisticRegression(solver="lbfgs"), alpha=.1),
    'GES-RB-LR': GES(base_clf=LogisticRegression(solver="lbfgs"), alpha=.1, beta=.1),
    'NB': GaussianNB(),
    'RS-NB': GES(base_clf=GaussianNB(), num_iter=0),
    'GES-NB': GES(base_clf=GaussianNB()),
    'GES-RA-NB': GES(base_clf=GaussianNB(), alpha=.1),
    'GES-RB-NB': GES(base_clf=GaussianNB(), alpha=.1, beta=.1),
}
folds = 5

# Load cube
results = []
datasets = h.datasets()
for dataset in datasets:
    score = np.load("results/%s.npy" % dataset)
    results.append(score)
results = np.array(results)
print(results.shape)

for m, metric in enumerate(metrics):
    print("Results for %s score" % metric)
    m_results = results[:, :, :, m]
    mean_scores = np.mean(m_results, axis=1)
    #print(mean_scores)

    np.savetxt("output/%s.csv" % metric, mean_scores, fmt="%.3f", delimiter=",")
    f = open('output/%s.tex' % metric, 'wt', encoding='utf-8')

    # Dependency
    for d, dataset in enumerate(datasets):
        row = mean_scores[d]

        # If is dependent on best
        bold = np.zeros(len(classifiers)).astype(int)
        better_than = []

        # LR
        best_id = np.argmax(row[:5])
        for i in range(5):
            local = []
            a = m_results[d, :, i]
            if i == best_id:
                bold[i] = 1
            for j in range(5):
                if i != j:
                    b = m_results[d, :, j]
                    c = test(a, b)
                    p = c.pvalue
                    if np.sum(a - b) == 0 and j == best_id:
                        bold[i] = 1
                    if p <= p_t and row[i] > row[j]:
                        local.append(j + 1)
                    if p > p_t and j == best_id:
                        bold[i] = 1

            better_than.append(local)

        # NB
        best_id = np.argmax(row[5:]) + 5
        for i in range(5, 10):
            local = []
            a = m_results[d, :, i]
            if i == best_id:
                bold[i] = 1
            for j in range(5, 10):
                if i != j:
                    b = m_results[d, :, j]
                    c = test(a, b)
                    p = c.pvalue
                    if np.sum(a - b) == 0 and j == best_id:
                        bold[i] = 1
                    if p <= p_t and row[i] > row[j]:
                        local.append(chr(97 + j - 5))
                    if p > p_t and j == best_id:
                        bold[i] = 1

            better_than.append(local)

        z = dataset.replace("_", "-")

        a = "\\emph{%s} & " % z + \
            " & ".join(["%s%.3f" % ("\\bfseries " if bold[i]
                                    == 1 else "", score) for i, score in enumerate(row)]) + " \\\\\n"
        #print(a)

        f.write(a)

        b = "& "+ " & ".join([",".join("-" if len(group) == 0 else ["%i" % one if idx_group < 5 else "%s" % one
                       for idx_one, one in enumerate(group)])
             for idx_group, group in enumerate(better_than)]) + " \\\\\n"
        f.write(b)
        #print(b)
