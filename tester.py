import argparse
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os
from FDA import FDA

parser = argparse.ArgumentParser(description="Flexible Discriminant Analysis")
parser.add_argument("--method", default=None, type=str, help="type of ambiguity set")
parser.add_argument("--rho", default=0, nargs="+", type=float, help="radius of ambiguity set")
parser.add_argument("--cv", default=1, type=int, help="number of folds for validation")
parser.add_argument("--repeat", default=1, type=int, help="number of test/train split")
parser.add_argument("--dataset", default=None, type=str, help="datafile address")
args = parser.parse_args()

DIR_SAVE = os.path.join(os.environ["HOME"], "Soroosh/results_full")
if not os.path.exists(DIR_SAVE):
    os.makedirs(DIR_SAVE)


def compute_scores(clf, y_ture, y_prob):
    y_prob_1 = y_prob[:, :clf.n_class_]
    y_prob_2 = y_prob[:, clf.n_class_:]
    fpr1, tpr1, thresholds1 = roc_curve(y_ture,  y_prob_1[:, 1], pos_label=1)
    fpr2, tpr2, thresholds2 = roc_curve(y_ture,  y_prob_2[:, 1], pos_label=1)
    return np.array([accuracy_score(y_ture, clf.labels_[y_prob_1.argmax(1)]),
                     accuracy_score(y_ture, clf.labels_[y_prob_2.argmax(1)]),
                     auc(fpr1, tpr1),
                     auc(fpr2, tpr2)])


def val_test_score(clf, X_tr, y_tr, X_te, y_te):
    skf = StratifiedKFold(n_splits=args.cv)
    skf.get_n_splits(X_tr, y_tr)
    scores_val = []
    for train_index, val_index in skf.split(X_tr, y_tr):
        X_train, X_val = X_tr[train_index], X_tr[val_index]
        y_train, y_val = y_tr[train_index], y_tr[val_index]
        y_val_prob = clf.fit(X_train, y_train).predict_proba(X_val)
        scores_val.append(compute_scores(clf, y_val, y_val_prob))
    y_prob = clf.fit(X_tr, y_tr).predict_proba(X_te)
    scores_te = compute_scores(clf, y_te, y_prob)
    return np.append(np.mean(scores_val, axis=0), scores_te)


def main():
    f_name = args.dataset[11:-4]
    if args.method is not None:
        f_name += "_" + args.method
    if hasattr(args.rho, '__iter__'):
        for rho in args.rho:
            f_name += "_" + str(rho)
    f_name = os.path.join(DIR_SAVE, f_name + ".csv")
    print("training & testing the {} dataset with {} method and rho {}".format(
        args.dataset[11:-4], args.method, args.rho))
    if os.path.exists(f_name):
        print("the model is already trained")
    else:
        clf = FDA(rule="fda", method=args.method, rho=args.rho)
        data = load_svmlight_file(args.dataset)
        X_data = data[0]
        y_data = data[1]
        labels = np.unique(y_data)
        y_data[y_data == labels[0]] = 0
        y_data[y_data == labels[1]] = 1
        scores = []
        for i in range(args.repeat):
            print("running iteration ", i + 1)
            X_train, X_test, y_train, y_test = train_test_split(
                X_data.toarray(), y_data, test_size=0.25, random_state=1000+i)
            scores.append(val_test_score(clf, X_train, y_train, X_test, y_test))
        np.savetxt(f_name, 100 * np.array(scores), fmt="%0.2f", delimiter=",")

if __name__ == "__main__":
    main()
