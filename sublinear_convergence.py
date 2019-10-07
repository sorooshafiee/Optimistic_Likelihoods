import os
import time
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

cwd = os.getcwd()
DIR_CSV = os.path.join(cwd, "csv")
DIR_FIG = os.path.join(cwd, "graph")
time_csv = os.path.join(DIR_CSV, "ex_time.csv")
iter_csv = os.path.join(DIR_CSV, "ex_iter.csv")
conv_csv = os.path.join(DIR_CSV, "conv.csv")
rc('font', family='serif')
rc('text', usetex=True)
rc('xtick',labelsize=14)
rc('ytick',labelsize=14)

all_n = np.arange(1,11) * 10
M = 1
sig = 10
thr = 1e-4
rho = 1e-2
max_run = 1001
repeat = 100
load_data = True


def inner(X, Y):
    return X.ravel() @ Y.ravel()


def func_mat(mat, func):
    """ compute function of a matrix """
    mat = 0.5 * (mat + mat.T)
    d, U = np.linalg.eigh(np.real(mat))
    return np.dot(func(d) * U, U.T)


def plot_with_shade(x, y, xlabel, ylabel, fname, loglog=False):
    y_mean = np.mean(y, axis=1)
    y_max = np.max(y, axis=1)
    y_min = np.min(y, axis=1)
    fig, ax = plt.subplots(1)
    ax.plot(x, y_mean, lw=4, color=[0, 0.4470, 0.7410])
    ax.fill_between(x, y_max, y_min, facecolor=[0, 0.4470, 0.7410], alpha=0.2)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True) #, which="both")
    ax.set_xlim(x.min(), x.max())
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig.savefig(fname, format='pdf', dpi=1000)
    return fig, ax


def proj_FR(cov_sqrt, prec_sqrt, prec, cov_new, rho):
    """ projection to FR ball """
    sigma = np.real(np.linalg.eigvals(prec @ cov_new))
    log_sigma = np.log(sigma) ** 2
    L = log_sigma.sum() ** 0.5
    if L <= rho:
        d, V = np.linalg.eigh(cov_new)
        return cov_new, np.abs(d), V
    else:
        t = rho / L
        frac_pow = func_mat(prec_sqrt @ cov_new @ prec_sqrt, lambda x: x ** t)
        proj_cov = cov_sqrt @ frac_pow @ cov_sqrt
        d, V = np.linalg.eigh(proj_cov)
        return proj_cov, np.abs(d), V


def PGD(S, cov, rho, criterion=None):
    """ PGD algorithm """
    n = cov.shape[0]
    # Implement the PGD algorithm
    d_k, V_k = np.linalg.eigh(cov)
    s_k, U_k = np.linalg.eigh(cov)
    t_k = np.linalg.eigh(S)[0]
    cov_sqrt = np.dot(np.sqrt(d_k) * V_k, V_k.T)
    prec_sqrt = np.dot(np.sqrt(1 / d_k) * V_k, V_k.T)
    prec = np.dot((1 / d_k) * V_k, V_k.T)
    obj_k = []
    d_min = np.min(d_k)
    t_max = np.max(t_k)
    Lambda = np.sqrt(n) * np.exp(2 * rho) * np.maximum(1, np.abs(1 - np.exp(rho) * t_max/ d_min)) / (d_min ** 2)
    e_bar = np.sqrt(2 * np.sqrt(2) * np.tanh(np.sqrt(2) * rho)) / Lambda
    iter_k = max_run
    for k in range(max_run):
        # Update sequence X
        X_sqrt_k = np.dot(np.sqrt(s_k) * U_k, U_k.T)
        X_inv_sqrt_k = np.dot(np.sqrt(1 / s_k) * U_k, U_k.T)
        g_k = np.eye(n) - X_inv_sqrt_k @ S @ X_inv_sqrt_k
        eta_k = e_bar / np.sqrt(k + 1)
        exp = func_mat(-eta_k * g_k, lambda x: np.exp(x))
        X_k_half = X_sqrt_k @ exp @ X_sqrt_k
        X_k, s_k, U_k = proj_FR(cov_sqrt, prec_sqrt, prec, X_k_half, rho)
        # Update sequence cov
        cov_sqrt_k = np.dot(np.sqrt(d_k) * V_k, V_k.T)
        prec_sqrt_k = np.dot(np.sqrt(1 / d_k) * V_k, V_k.T)
        prec_k = np.dot((1 / d_k) * V_k, V_k.T)
        t = 1 / (k + 2)
        prec_k_pow = func_mat(prec_sqrt_k @ X_k @ prec_sqrt_k,
                              lambda x: x ** t)
        cov_k = cov_sqrt_k @ prec_k_pow @ cov_sqrt_k
        d_k, V_k = np.linalg.eigh(cov_k)
        # Check stopping criteria
        obj_k.append(np.asscalar(inner(S, prec_k) + np.log(d_k).sum()))
        res = np.absolute((obj_k[-1] - obj_k[-2]) / obj_k[-2]) if k > 2 else np.inf
        if criterion is not None and res < criterion:
            iter_k = k
            break
    return np.dot(d_k * V_k, V_k.T), obj_k, iter_k


def main():
    if load_data \
            and os.path.isfile(time_csv)\
            and os.path.isfile(iter_csv)\
            and os.path.isfile(conv_csv):
        ex_time = np.genfromtxt(time_csv, delimiter=',')
        ex_iter = np.genfromtxt(iter_csv, delimiter=',')
        conv = np.genfromtxt(conv_csv, delimiter=',')
    else:
        ex_time = []
        ex_iter = []
        conv = []
        for n in all_n:
            print(n)
            time_ = []
            iter_ = []
            for r in range(repeat):
                A = np.random.normal(0, 1, (n, n))
                d, V = np.linalg.eigh(A)
                d = np.random.uniform(0.1*sig, sig, n)
                cov = np.dot(d * V, V.T)
                X = np.random.normal(0, sig, (n, M)) / np.sqrt(M)
                S = X @ X.T
                t = time.time()
                cov, obj, it = PGD(S, cov, rho * np.sqrt(n), thr)
                time_.append(time.time() - t)
                iter_.append(it)
            ex_time.append(time_)
            ex_iter.append(iter_)
        for r in range(repeat):
            print(r)
            cov = np.eye(all_n.max())
            X = np.random.normal(0, sig, (all_n.max(), M)) / np.sqrt(M)
            S = X @ X.T
            cov, obj, it = PGD(S, cov, rho * np.sqrt(all_n.max()))
            conv.append(obj)
        ex_time = np.array(ex_time)
        ex_iter = np.array(ex_iter)
        conv = np.array(conv)
        np.savetxt(time_csv, ex_time, fmt="%0.5f", delimiter=",")
        np.savetxt(iter_csv, ex_iter, fmt="%0.5f", delimiter=",")
        np.savetxt(conv_csv, conv, fmt="%0.5f", delimiter=",")

    plot_with_shade(all_n, ex_time, 'Dimension ($n$)', 'Execution time (s)',
                    os.path.join(DIR_FIG, "time.pdf"))
    plot_with_shade(all_n, ex_iter, 'Dimension ($n$)', '\# of iterations',
                    os.path.join(DIR_FIG, "iteration.pdf"))
    change = np.diff(conv)
    err = np.abs(change / conv[:, 1:])
    plot_with_shade(np.arange(1, max_run), err.T, 'Iteration', '$ | [L(\Sigma_{k+1}) - L(\Sigma_{k})] / L(\Sigma_{k+1}) | $',
                    os.path.join(DIR_FIG, "convergence.pdf"), True)


if __name__ == "__main__":
    main()






