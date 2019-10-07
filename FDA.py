import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf, GraphicalLasso, graphical_lasso


def _soft_max(log_x):
    """ compute softmax operator """
    x = np.exp(log_x - log_x.max())
    return x / x.sum()


def _comp_prob_rule(x, mean, prec, logdet, priors, rule):
    """ compute probability from likelihood function and discriminant rule """
    n_class = len(mean)
    log_likelihood = np.zeros(n_class)
    for n_c in range(n_class):
        x_bar = x - mean[n_c]
        log_likelihood[n_c] = - 0.5 * x_bar @ prec[n_c] @ x_bar - 0.5 * logdet[n_c]
    if rule.lower() == "da":
        return _soft_max(log_likelihood)
    elif rule.lower() == "qda":
        return _soft_max(log_likelihood + np.log(priors))
    elif rule.lower() == "fda":
        return np.append(_soft_max(log_likelihood), _soft_max(log_likelihood + np.log(priors)))


def _func_mat(mat, func):
    """ compute function of a matrix """
    d, U = np.linalg.eigh(mat)
    return np.dot(func(np.abs(d)) * U, U.T)


class FDA(BaseEstimator, ClassifierMixin):
    """ Flexible Discriminant Analysis """

    def __init__(
            self,
            rule='QDA',
            method=None,
            priors=None,
            adaptive=True,
            rho=0,
            tol=1e-8,
            verbose=False
    ):
        """Copy params to object properties"""
        self.rule = rule.lower()
        self.method = method if method is None else method.lower()
        self.priors = priors
        self.adaptive = adaptive
        self.rho = rho
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the QDA to the training data"""
        methods = [None, "fr", "kl", "mean", "wass", "reg", "freg", "sparse"]
        rules = ["qda", "da", "fda"]
        if self.method not in methods:
            raise ValueError("method must be in {}; got (method={})".format(methods, self.method))
        if self.rule not in rules:
            raise ValueError("rule must be in {}; got (rule={})".format(rules, self.rule))
        X, y = check_X_y(X, y)

        self.labels_, self.n_samples_ = np.unique(y, return_counts=True)
        self.n_class_ = self.labels_.size
        n_samples, self.n_features_ = X.shape

        self.rho_ = np.array([self.rho]).ravel()
        if self.rho_.size == 1:
            self.rho_ = self.rho_[0] * np.ones(self.n_class_)
        if self.adaptive:
            self.rho_ *= np.sqrt(self.n_features_)
        if self.priors is None:
            self.priors_ = self.n_samples_ / n_samples
        else:
            self.priors_ = self.priors
        self.mean_ = []
        self.covariance_ = []
        self.cov_sqrt_ = []
        self.prec_ = []
        self.prec_sqrt_ = []
        self.logdet_ = []
        self.rotations_ = []
        self.scalings_ = []
        for n_c, label in enumerate(self.labels_):
            mask = (y == label)
            X_c = X[mask, :]
            X_c_mean = np.mean(X_c, 0)
            X_c_bar = X_c - X_c_mean
            U, s, Vt = np.linalg.svd(X_c_bar, full_matrices=False)
            s2 = (s ** 2) / (len(X_c_bar) - 1)
            self.mean_.append(X_c_mean)
            if self.method == 'reg':
                s2 += self.rho_[n_c]
                inv_s2 = 1 / s2
            elif self.method in ['fr', 'kl', 'mean', 'freg']:
                sc = StandardScaler()
                X_c_ = sc.fit_transform(X_c)
                cov_c = ledoit_wolf(X_c_)[0]
                cov_c = sc.scale_[:, np.newaxis] * cov_c * sc.scale_[np.newaxis, :]
                s2, V = np.linalg.eigh(cov_c)
                s2 = np.abs(s2)
                inv_s2 = 1 / s2
                Vt = V.T
            elif self.method == 'sparse':
                try:
                    cov_c = GraphicalLasso(alpha=self.rho_[n_c]).fit(X_c_bar)
                    cov_c = cov_c.covariance__
                except:
                    tol = self.tol * 1e6
                    cov_c = graphical_lasso(np.dot(((1 - tol) * s2 + tol) * Vt.T, Vt), self.rho_[n_c])[0]
                s2, V = np.linalg.eigh(cov_c)
                s2 = np.abs(s2)
                inv_s2 = 1 / s2
                Vt = V.T
            elif self.method == 'wass':
                f = lambda gamma: gamma * (self.rho_[n_c] ** 2 - 0.5 * np.sum(s2)) - self.n_features_ + \
                                  0.5 * (np.sum(np.sqrt((gamma ** 2) * (s2 ** 2) + 4 * s2 * gamma)))
                lb = 0
                gamma_0 = 0
                ub = np.sum(np.sqrt(1 / (s2 + self.tol) )) / self.rho_[n_c]
                f_ub = f(ub)
                for bsect in range(100):
                    gamma_0 = 0.5 * (ub + lb)
                    f_gamma_0 = f(gamma_0)
                    if f_ub * f_gamma_0 > 0:
                        ub = gamma_0
                        f_ub = f_gamma_0
                    else:
                        lb = gamma_0
                    if abs(ub - lb) < self.tol:
                        break
                inv_s2 = gamma_0 * (1 - 2 / (1 + np.sqrt(1 + 4 / (gamma_0 * (s2 + self.tol)))))
                s2 = 1 / (inv_s2 + self.tol)
            else:
                s2 += self.tol
                inv_s2 = 1 / s2
            self.covariance_.append(np.dot(s2 * Vt.T, Vt))
            self.cov_sqrt_.append(np.dot(np.sqrt(s2) * Vt.T, Vt))
            self.prec_.append(np.dot(inv_s2 * Vt.T, Vt))
            self.prec_sqrt_.append(np.dot(np.sqrt(inv_s2) * Vt.T, Vt))
            self.logdet_.append(np.log(s2).sum())
            self.rotations_.append(Vt)
            self.scalings_.append(s2)
        return self

    def predict(self, X):
        """ Predict the labels of a set of features """
        prob = self.predict_proba(X)
        if self.rule == 'fda':
            prob_1 = prob[:, :self.n_class_]
            prob_2 = prob[:, self.n_class_:]
            return np.vstack((self.labels_[prob_1.argmax(1)], self.labels_[prob_2.argmax(1)]))
        else:
            return self.labels_[prob.argmax(1)]

    def predict_proba(self, X):
        """ Estimate the probability for the each class """
        X = check_array(X, dtype=np.float64)
        prob = []
        for i, x in enumerate(X):
            prob.append(self._max_like_est(x))
        return np.array(prob)

    def _max_like_est(self, x):
        """ Maximum Likelihood Estimation for the each class """
        if self.method in [None, "wass", "reg", "freg", "sparse"]:
            return _comp_prob_rule(x, self.mean_, self.prec_,
                                   self.logdet_, self.priors_, self.rule)
        elif "kl" == self.method.lower():
            return self._KL(x)
        elif "mean" == self.method.lower():
            return self._mean(x)
        elif "fr" == self.method.lower():
            return self._FR(x)

    def _KL(self, x):
        """ """
        gamma_0 = np.array([1])
        n = self.n_features_
        prec_star = []
        logdet_star = []
        for n_c in range(self.n_class_):
            x_bar = x - self.mean_[n_c]
            x_vec = x_bar[:, np.newaxis]
            x_norm = x_bar @ self.prec_[n_c] @ x_bar
            rho = self.rho_[n_c] #** 2
            f = lambda gamma: gamma * rho + n * (1 + gamma) * np.log(1 + gamma) - \
                              (1 + gamma) * np.log(gamma + x_norm) + \
                              (1 - n) * (1 + gamma) * np.log(gamma)
            g = lambda gamma: rho + n * (np.log(1 + gamma) + 1) - \
                              (gamma + 1) / (gamma + x_norm) - np.log(gamma + x_norm) + \
                              (1 - n) * (np.log(gamma) + 1 / gamma + 1)
            res = minimize(f, gamma_0, method='L-BFGS-B', jac=g,
                           bounds=((1e-4, np.inf),), tol=1e-6)
            gamma_star = res.x[0]
            pr_c = self.prec_[n_c]
            prec_star.append(
                (1 + 1 / gamma_star) * \
                (pr_c - (pr_c @ x_vec @ x_vec.T @ pr_c) / (gamma_star + x_norm)))
            logdet_star.append(
                self.logdet_[n_c] + (n - 1) * np.log(gamma_star) - \
                n * np.log(1 + gamma_star) + np.log(gamma_star + x_norm))
        return _comp_prob_rule(x, self.mean_, prec_star, logdet_star, self.priors_, self.rule)

    def _mean(self, x):
        """ """
        gamma_0 = np.array([100])
        mean_star = []
        for n_c in range(self.n_class_):
            mu_hat = self.mean_[n_c]
            rho = self.rho_[n_c] #** 2
            mu_norm = mu_hat @ self.prec_[n_c] @ mu_hat
            f = lambda gamma: gamma * (rho - mu_norm) + (x + gamma * mu_hat) \
                              @ self.prec_[n_c] @ (x + gamma * mu_hat) / (1 + gamma)
            g = lambda gamma: rho - mu_norm + ((2 + gamma) * mu_hat - x) \
                              @ self.prec_[n_c] @ (x + gamma * mu_hat) / ((1 + gamma) ** 2)
            res = minimize(f, gamma_0, method='L-BFGS-B', jac=g,
                           bounds=((1e-4, np.inf),), tol=1e-6)
            gamma_star = res.x[0]
            mean_star.append((x + gamma_star * mu_hat) / (1 + gamma_star))
        return _comp_prob_rule(x, mean_star, self.prec_, self.logdet_, self.priors_, self.rule)

    def _FR(self, x):
        """ """
        prec_star = []
        logdet_star = []
        for n_c in range(self.n_class_):
            x_bar = x - self.mean_[n_c]
            d_star, V_star = self._geodesic_descent(x_bar, n_c)
            inv_d_star = 1 / d_star
            prec_star.append(np.dot(inv_d_star * V_star, V_star.T))
            logdet_star.append(np.log(np.abs(d_star)).sum())
        return _comp_prob_rule(x, self.mean_, prec_star, logdet_star, self.priors_, self.rule)

    def _proj_FR_(self, cov_new, n_c):
        """ """
        cov_sqrt = self.cov_sqrt_[n_c]
        prec_sqrt = self.prec_sqrt_[n_c]
        prec = self.prec_[n_c]
        rho_bar = self.rho_[n_c] #* np.sqrt(2)
        sigma = np.real(np.linalg.eigvals(prec @ cov_new))
        log_sigma = np.log(np.abs(sigma)) ** 2
        L = log_sigma.sum() ** 0.5
        if L <= rho_bar:
            d, V = np.linalg.eigh(cov_new)
            return cov_new, np.abs(d), V
        else:
            t = rho_bar / L
            frac_pow = _func_mat(prec_sqrt @ cov_new @ prec_sqrt, lambda x: x ** t)
            proj_cov = cov_sqrt @ frac_pow @ cov_sqrt
            d, V = np.linalg.eigh(proj_cov)
            return proj_cov, np.abs(d), V

    def _geodesic_descent(self, x_bar, n_c):
        """ """
        n = self.n_features_
        x_vec = x_bar[:, np.newaxis]
        # Implement the PGD algorithm
        d_k, V_k = self.scalings_[n_c].copy(), self.rotations_[n_c].copy()
        s_k, U_k = self.scalings_[n_c].copy(), self.rotations_[n_c].copy()
        prec = np.dot((1 / d_k) * V_k, V_k.T)
        obj_k = [np.asscalar(x_bar.T @ prec @ x_bar + np.log(np.abs(d_k)).sum())]
        d_min = np.min(d_k)
        t_max = x_bar.T @ x_bar
        Lambda = np.sqrt(n) * np.exp(2 * self.rho_[n_c]) * \
                 np.maximum(1, np.abs(1 - np.exp(self.rho_[n_c]) * t_max / d_min)) / (d_min ** 2)
        # e_bar = np.sqrt(2 * np.sqrt(2) * np.tanh(np.sqrt(2) * self.rho_[n_c])) / Lambda
        e_bar = 1e-1
        for k in range(100):
            # Update sequence X
            X_sqrt_k = np.dot(np.sqrt(s_k) * U_k, U_k.T)
            X_inv_sqrt_k = np.dot(np.sqrt(1 / s_k) * U_k, U_k.T)
            x_vec_k = X_inv_sqrt_k @ x_vec
            g_k = np.eye(n) - x_vec_k @ x_vec_k.T
            # This loop is ensuring numerical stability
            for i in range(100):
                eta_k = e_bar / np.sqrt(k + 1)
                with np.errstate(over='raise'):
                    try:
                        exp = _func_mat(-eta_k * g_k, lambda x: np.exp(x))
                        if np.linalg.norm(exp.ravel(), np.inf) <= 1e6:
                            break
                    except:
                        pass
                    e_bar /= 2
            X_k_half = X_sqrt_k @ exp @ X_sqrt_k
            X_k, s_k, U_k = self._proj_FR_(X_k_half, n_c)
            # Update sequence cov
            cov_sqrt_k = np.dot(np.sqrt(d_k) * V_k, V_k.T)
            prec_sqrt_k = np.dot(np.sqrt(1 / d_k) * V_k, V_k.T)
            t = 1 / (k + 2)
            prec_k_pow = _func_mat(prec_sqrt_k @ X_k @ prec_sqrt_k,
                                   lambda x: x ** t)
            cov_k = cov_sqrt_k @ prec_k_pow @ cov_sqrt_k
            d_k, V_k = np.linalg.eigh(cov_k)
            prec_k = np.dot((1 / d_k) * V_k, V_k.T)
            # Check stopping criteria
            obj_k.append(np.asscalar(x_bar.T @ prec_k @ x_bar + np.log(np.abs(d_k)).sum()))
            change = (obj_k[-1] - obj_k[-2]) / obj_k[-2]
            if change < 0 and np.absolute(change) < 1e-4 and k > 5:
                break

        return d_k, V_k
