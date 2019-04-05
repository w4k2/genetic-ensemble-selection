"""
Testing crossing and mutation probabilities.
"""
from GES import GES
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.metrics import geometric_mean_score
from metrics import f_score, bac
import numpy as np
import helper as h
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Processing parameters
classifiers = {
    'LR': LogisticRegression(solver="lbfgs"),
    'NB': GaussianNB()
}
metrics = {
    "f": f_score,
    "b": bac,
    "g": geometric_mean_score
}
density = 5
crossing = np.linspace(0, 0.1, density)
mutation = np.linspace(0, 0.02, density)
datasets = h.datasets()
folds = 3
num_iter = 100
pool_size = 20

for dataset in datasets:
    for m in metrics:
        for c_id in classifiers:

            metric = metrics[m]
            print("# Analyzing %s | %s" % (dataset, metric))
            X, y, _, _ = h.load_dataset(dataset)
            label_corrector = np.max(y) - 1

            # Prepare plot
            fig, ax = plt.subplots(density, density, figsize=(8,8))
            # Score storage
            scores = np.zeros((folds, density, density, num_iter, pool_size))
            qualities = np.zeros((folds, density, density, num_iter, pool_size))
            model_scores = np.zeros((folds, density, density))

            # Divide sets
            skf = StratifiedKFold(n_splits=folds, random_state=1337)
            for f, (train, test) in tqdm(
                enumerate(skf.split(X, y)), ascii=True, total=folds
            ):
                # Get subsets
                X_train, X_test = X[train], X[test]
                y_train, y_test = (y[train] - label_corrector, y[test] - label_corrector)

                for i, m_ in tqdm(enumerate(mutation), total=density, ascii=True):
                    for j, c_ in tqdm(enumerate(crossing), total=density, ascii=True):

                        # Initiate classifier
                        clf = GES(
                            num_iter=num_iter,
                            pool_size=pool_size,
                            p_mutation=m_,
                            p_crossing=c_,
                            metric=metric,
                            base_clf=classifiers[c_id]
                        )
                        clf.fit(X_train, y_train)

                        # Test classifier and save results

                        model_scores[f, i, j] = metric(
                            y_test, clf.predict(X_test)
                        )
                        scores[f, i, j, :, :] = clf.all_scores
                        # qualities[f, :, :] = clf.all_qualities

            final_qualities = np.copy(np.mean(scores[:, :, :, :, 0], axis=0))
            final_qualities = np.mean(final_qualities, axis=2)
            # final_qualities = np.mean(final_qualities, axis=2)

            final_qualities -= np.min(final_qualities)
            final_qualities /= np.max(final_qualities)

            score_qualities = np.copy(np.mean(model_scores, axis=0))
            score_qualities -= np.min(score_qualities)
            score_qualities /= np.max(score_qualities)

            # Plot
            for i, m_ in enumerate(mutation):
                for j, c_ in enumerate(crossing):
                    q_scores = scores[:, i, j, :, :]
                    # Fold-flatten scores
                    ff_scores = np.mean(q_scores, axis=0)
                    # ff_qualities = np.mean(qualities, axis=0)

                    best_scores = ff_scores[:, 0]
                    mean_scores = np.mean(ff_scores, axis=1)

                    # best_qualities = ff_qualities[:, 0]
                    # mean_qualities = np.mean(ff_qualities, axis=1)

                    # Plot
                    ax[i, j].plot(range(num_iter), best_scores, c="black")
                    ax[i, j].plot(range(num_iter), mean_scores, c="black", ls=":")
                    # plt.plot(range(epochs), best_qualities, c="red")
                    # plt.plot(range(epochs), mean_qualities, c="red", ls=":")

                    model_score = np.mean(model_scores[:, i, j])
                    ax[i, j].plot(
                        num_iter - 1,
                        model_score,
                        marker="o",
                        markersize=5,
                        color=(1, 1 - score_qualities[i, j], 1 - score_qualities[i, j], 1),
                        markeredgewidth=1,
                        markeredgecolor="black",
                    )

                    ax[i, j].set_ylim(0.4, 1)
                    if i != density - 1:
                        ax[i, j].set_xticks([])

                    if j != 0:
                        ax[i, j].set_yticks([])

                    ax[i, j].set_title(
                        "m %.1f - c%.2f - %.3f" % (m_ * 100, c_ * 100, model_score),
                        fontsize=8,
                    )

                    ax[i, j].set_facecolor(
                        (1 - final_qualities[i, j], 1, 1 - final_qualities[i, j], 1)
                    )
            plt.tight_layout()

            plt.tight_layout()
            plt.savefig("foo.png")
            plt.savefig("figures/mutation/%s_%s_%s.png" % (
            dataset, m, c_id))



            plt.close()
