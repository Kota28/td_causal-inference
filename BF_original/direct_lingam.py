"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import numpy as np
from sklearn.preprocessing import scale
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression
import itertools

from .base import _BaseLiNGAM


class DirectLiNGAM(_BaseLiNGAM):
    """Implementation of DirectLiNGAM Algorithm [1]_ [2]_

    References
    ----------
    .. [1] S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvärinen, Y. Kawahara, T. Washio, P. O. Hoyer and K. Bollen. 
       DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model. Journal of Machine Learning Research, 12(Apr): 1225--1248, 2011.
    .. [2] A. Hyvärinen and S. M. Smith. Pairwise likelihood ratios for estimation of non-Gaussian structural eauation models. 
       Journal of Machine Learning Research 14:111-152, 2013. 
    """

    def __init__(self, measure, mode, random_state=2022):
        """Construct a DirectLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        apply_prior_knowledge_softly : boolean, optional (default=False)
            If True, apply prior knowledge softly.
        measure : {'pwling', 'kernel'}, optional (default='pwling')
            Measure to evaluate independence: 'pwling' [2]_ or 'kernel' [1]_.
        """
        super().__init__(random_state)
        self._measure = measure
        self._mode = mode


    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        print(type(X))
        print(X)
        X = check_array(X)
        n_features = X.shape[1]

        # Causal discovery
        U = np.arange(n_features)
        K = []
        X_ = np.copy(X)
        if self._measure == 'kernel':
            X_ = scale(X_)
#             print("Var: " + str(np.var( X_[:, 1])))
#             print(np.cov( X_[:, 0],  X_[:, 1])[0, 1] / np.var( X_[:, 1]))

#         for _ in range(n_features):
#             if self._measure == 'kernel':
#                 m = self._search_causal_order_kernel(X_, U)
#             else:
#                 m = self._search_causal_order(X_, U)
#             for i in U:
#                 if i != m:
#                     X_[:, i] = self._residual(X_[:, i], X_[:, m])
#             K.append(m)
#             U = U[U != m]
            

        self._causal_order = self._search_causal_order_kernel(X_)
        return self._estimate_adjacency_matrix(X)

    def _extract_partial_orders(self, pk):
        """ Extract partial orders from prior knowledge."""
        path_pairs = np.array(np.where(pk == 1)).transpose()
        no_path_pairs = np.array(np.where(pk == 0)).transpose()

        # Check for inconsistencies in pairs with path
        check_pairs = np.concatenate([path_pairs, path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            if len(pairs[counts > 1]) > 0:
                raise ValueError(
                    f'The prior knowledge contains inconsistencies at the following indices: {pairs[counts>1].tolist()}')

        # Check for inconsistencies in pairs without path.
        # If there are duplicate pairs without path, they cancel out and are not ordered.
        check_pairs = np.concatenate([no_path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            check_pairs = np.concatenate([no_path_pairs, pairs[counts > 1]])
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            no_path_pairs = pairs[counts < 2]

        check_pairs = np.concatenate([path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) == 0:
            # If no pairs are extracted from the specified prior knowledge, 
            # discard the prior knowledge.
            return None

        pairs = np.unique(check_pairs, axis=0)
        return pairs[:, [1, 0]]  # [to, from] -> [from, to]

    def _residual(self, xi, xj):
        """The residual when xi is regressed on xj."""
        return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj

    def _entropy(self, u):
        """Calculate entropy using the maximum entropy approximations."""
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        print("entropy")
        return (1 + np.log(2 * np.pi)) / 2 - \
            k1 * (np.mean(np.log(np.cosh(u))) - gamma)**2 - \
            k2 * (np.mean(u * np.exp((-u**2) / 2)))**2

    def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
        """Calculate the difference of the mutual informations."""
        return (self._entropy(xj_std) + self._entropy(ri_j / np.std(ri_j))) - \
               (self._entropy(xi_std) + self._entropy(rj_i / np.std(rj_i)))

    def _search_candidate(self, U):
        """ Search for candidate features """
        # If no prior knowledge is specified, nothing to do.

        # Find exogenous features
        Uc = []
        for j in U:
            index = U[U != j]

        # Find endogenous features, and then find candidate features
        if len(Uc) == 0:
            U_end = []
            for j in U:
                index = U[U != j]
                
            # Find sink features (original)
            for i in U:
                index = U[U != i]
              
            Uc = [i for i in U if i not in set(U_end)]

        # make V^(j)
        Vj = []
        for i in U:
            if i in Uc:
                continue
        return Uc, Vj

    def _search_causal_order(self, X, U):
        """Search the causal ordering."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        M_list = []
        for i in Uc:
            M = 0
            for j in U:
                if i != j:
                    xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                    xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                    ri_j = xi_std if i in Vj and j in Uc else self._residual(
                        xi_std, xj_std)
                    rj_i = xj_std if j in Vj and i in Uc else self._residual(
                        xj_std, xi_std)
                    M += np.min([0, self._diff_mutual_info(xi_std,
                                                           xj_std, ri_j, rj_i)])**2
            M_list.append(-1.0 * M)
        return Uc[np.argmax(M_list)]

    def _mutual_information(self, x1, x2, param):
        """Calculate the mutual informations."""
        kappa, sigma = param
        n = len(x1)
        X1 = np.tile(x1, (n, 1))
        K1 = np.exp(-1/(2*sigma**2) * (X1**2 + X1.T**2 - 2*X1*X1.T))
        X2 = np.tile(x2, (n, 1))
        K2 = np.exp(-1/(2*sigma**2) * (X2**2 + X2.T**2 - 2*X2*X2.T))

        tmp1 = K1 + n*kappa*np.identity(n)/2
        tmp2 = K2 + n*kappa*np.identity(n)/2
        K_kappa = np.r_[np.c_[tmp1 @ tmp1, K1 @ K2],
                        np.c_[K2 @ K1, tmp2 @ tmp2]]
        D_kappa = np.r_[np.c_[tmp1 @ tmp1, np.zeros([n, n])],
                        np.c_[np.zeros([n, n]), tmp2 @ tmp2]]

        sigma_K = np.linalg.svd(K_kappa, compute_uv=False)
        sigma_D = np.linalg.svd(D_kappa, compute_uv=False)
#         print("you are using a kernel-based estimator")

        return (-1/2)*(np.sum(np.log(sigma_K)) - np.sum(np.log(sigma_D)))

#     def _search_causal_order_kernel(self, X, U):
#         """Search the causal ordering by kernel method."""
#         Uc, Vj = U, []
# #         K = []
#         A = [i for i in range(len(U))]
#         if len(Uc) == 1:
#             return Uc[0]

#         if X.shape[0] > 1000:
#             param = [2e-3, 0.5]
#         else:
#             param = [2e-2, 1.0]

#         Tkernels = []
# #         print(U == Uc)
#         for j in Uc: # x_j will be an explanatory variable
#             Tkernel = 0
# #             print(Uc)
#             result = list(set(A) - set(Uc))
# #             print(result)
#             for i in U:
#                 if i != j:
# #                     print(np.cov( X[:, i],  X[:, j])[0, 1] / np.var( X[:, j]))
#                     ri_j = self._residual(X[:, i], X[:, j])
# #                     print("-------------------")
# #                     print(i, j)
#                     print(ri_j)
# #                     print(type(ri_j))
# #                     print(len(ri_j))
# #                     print("-------------------")
#                     Tkernel += self._mutual_information(X[:, j], ri_j, param)
# #                     print(Tkernel)
#             Tkernels.append(Tkernel)
# #             print(Tkernels)
# #         K.append(Uc[np.argmin(Tkernels)])
# #         print(K)
#         return Uc[np.argmin(Tkernels)]

    
    def _BF_res(self, d, m, X):
#         def residual(xi, xj):
#             return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj
        if m == 1:
            return self._residual(X[:, d-1], X[:, 0]) 
        else:
            return self._residual(self._BF_res(d, m-1, X), X[:, m-1])
    
    def _multiple_BF_res(self, d, m, X):
        if m == 1:
            return self._residual(X[:, d-1], X[:, 0])
        else:
            explanatory = [i for i in range(m)]
            return self._multiple_residual(X[:, d-1], X[:, explanatory])
    
    def _pre_loss(self, dimension, d_all, X):
        
        if X.shape[0] > 1000:
            param = [2e-3, 0.5]
        else:
            param = [2e-2, 1.0]
            
        ans = 0.0
        if self._mode == "original":
            if dimension == 1:
                for i in range(2, d_all+1):
                    ans += self._mutual_information(X[:, 0], self._BF_res(i, 1, X), param)
            else:
                for i in range(dimension+1, d_all+1):
                    ans += self._mutual_information(self._BF_res(dimension, dimension-1, X), self._BF_res(i, dimension, X), param)
        else:
            if dimension == 1:
                for i in range(2, d_all+1):
                    ans += self._mutual_information(X[:, 0], self._multiple_BF_res(i, 1, X), param)
            else:
                for i in range(dimension+1, d_all+1):
                    ans += self._mutual_information(self._multiple_BF_res(dimension, dimension-1, X), self._multiple_BF_res(i, dimension, X), param)
        
        return ans
    
    def _LOSS(self, d, X):
        res = 0.0
        for i in range(1, d):
            res += self._pre_loss(i, d, X)
        return res
    
    def _multiple_residual(self, xi, xj):
        """The residual when xi is regressed on xj. xi = y, xj = x"""
        model_lr = LinearRegression()
        model_lr.fit(xj, xi)
        res = np.array(xi - model_lr.predict(xj))
        return res
    
    def _search_causal_order_kernel(self, X):
        """Search the causal ordering by kernel method."""
#         if len(U) == 1:
#             return U[0]
        Tkernels = []
        d = X.shape[1]
        colchanged_X = np.zeros((len(X), d))
        arr = [i for i in range(d)]
        change_dict = list(itertools.permutations(arr))
        for i in range(len(change_dict)):
            for j in range(len(change_dict[i])):
                colchanged_X[:, j] = X[:, change_dict[i][j]]
            
            Tkernels.append(self._LOSS(d, colchanged_X))
        index = np.argmin(Tkernels)
        return list(change_dict[index])
            
     