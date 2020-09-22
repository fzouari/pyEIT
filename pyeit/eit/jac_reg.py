# coding: utf-8
# pylint: disable=invalid-name, no-member, too-many-arguments
# pylint: disable=too-many-instance-attributes, too-many-locals
# pylint: disable=arguments-differ
""" dynamic EIT solver using JAC """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.linalg as la
from numpy.linalg import multi_dot

from .base import EitBase


class JAC(EitBase):
    """ A sensitivity-based EIT imaging class """

    def setup(self, p=0.20, lamb=0.001, method='kotre', W=None, Wx=None, xp=None):
        """
        JAC, default file parser is 'std'

        Parameters
        ----------
        p, lamb: float
            JAC parameters
        method: str
            regularization methods
        """
        jac = self.J
        if W is None:
            W = np.eye(jac.shape[0])
        if Wx is None:
            Wx = np.eye(jac.shape[1])
        if xp is None:
            xp = np.zeros((jac.shape[1]))

        # passing imaging parameters
        self.params = {
            'p': p,
            'lamb': lamb,
            'method': method,
            'W': W,
            'Wx': Wx,
            'xp': xp
        }
        # pre-compute H0 for dynamical imaging
        # H = (J.T*J + R)^(-1) * J.T
        self.H = h_matrix(self.J, p, lamb, method, W)

        # pre-compute Q for dynamical imaging
        # Q = (J.T*W*J + R)^(-1)
        self.Q, r_mat = qr_matrix(self.J, p, lamb, W, Wx, method=method)

        self.params['Wx'] = np.dot(r_mat, Wx)

    def solve(self, v1, v0, normalize=False):
        """ dynamic solve_eit

        Parameters
        ----------
        v1: NDArray
            current frame
        v0: NDArray, optional
            referenced frame, d = H(v1 - v0)
        normalize: Boolean
            true for conducting normalization

        Returns
        -------
        ds: NDArray
            complex-valued NDArray, changes of conductivities
        """
        if normalize:
            dv = self.normalize(v1, v0)
        else:
            dv = (v1 - v0)
        # s = -Hv

        jac = self.J
        q_mat = self.Q
        W = self.params['W']
        Wx = self.params['Wx']
        lamb = self.params['lamb']
        xp = self.params['xp']

        h_mat = multi_dot([q_mat, jac.transpose(), W])
        p_mat = np.dot(q_mat, Wx)
        ds = -np.dot(h_mat, dv) + lamb*np.dot(p_mat, xp)

        return ds

    def solve_P(self, v1, v0, normalize=False):
        """ dynamic solve_eit

        Parameters
        ----------
        v1: NDArray
            current frame
        v0: NDArray, optional
            referenced frame, d = H(v1 - v0)
        normalize: Boolean
            true for conducting normalization

        Returns
        -------
        (xi, eta): Points on L-curve
        """
        if normalize:
            dv = self.normalize(v1, v0)
        else:
            dv = (v1 - v0)
        # s = -Hv

        jac = self.J
        h_mat = multi_dot([self.Q, jac.transpose(), self.params['W']])
        p_mat = np.dot(self.Q, self.params['Wx'])

        ds = -np.dot(h_mat, dv) + self.params['lamb']*np.dot(p_mat, self.params['xp'])

        Ax_minus_y = np.dot(-jac, ds) - dv
        x_minus_xp = ds - self.params['xp']

        xi = np.log10(multi_dot([np.conjugate(Ax_minus_y).transpose(), self.params['W'], Ax_minus_y]))
        eta = np.log10(multi_dot([np.conjugate(x_minus_xp).transpose(), self.params['Wx'], x_minus_xp]))
        # xi = multi_dot([np.conjugate(Ax_minus_y).transpose(), self.params['W'], Ax_minus_y])
        # eta = multi_dot([np.conjugate(x_minus_xp).transpose(), self.params['Wx'], x_minus_xp])

        return (xi, eta)

    def solve_G(self, v1, v0, normalize=False):
        """ dynamic solve_eit

        Parameters
        ----------
        v1: NDArray
            current frame
        v0: NDArray, optional
            referenced frame, d = H(v1 - v0)
        normalize: Boolean
            true for conducting normalization

        Returns
        -------
        G: GCV value
        """
        if normalize:
            dv = self.normalize(v1, v0)
        else:
            dv = (v1 - v0)

        b = dv
        L1 = np.linalg.cholesky(self.params['W'])
        L1_T = L1.transpose()
        A = -self.J

        A_prime = np.dot(L1_T,A)
        b_prime = np.dot(L1_T,dv-np.dot(A, self.params['xp']))

        AI_mat = multi_dot([self.Q, A.transpose(), L1])
        xreg_prime = np.dot(AI_mat, b_prime)

        Ax_minus_y = np.dot(A_prime, xreg_prime) - b_prime  # A'x'-b'
        numer = np.dot(np.conjugate(Ax_minus_y).transpose(),Ax_minus_y)

        Im = np.eye(A_prime.shape[0])
        denom = np.trace(Im - np.dot(A_prime,AI_mat))**2

        G = numer/denom

        return G

    def map(self, v):
        """ return Hv """
        return -np.dot(self.H, v)

    def solve_gs(self, v1, v0):
        """ solving by weighted frequency """
        a = np.dot(v1, v0) / np.dot(v0, v0)
        dv = (v1 - a*v0)
        ds = -np.dot(self.H, dv)
        # return average epsilon on element
        return ds

    def jt_solve(self, v1, v0, normalize=True):
        """
        a 'naive' back projection using the transpose of Jac.
        This scheme is the one published by kotre (1989):

        [1] Kotre, C. J. (1989).
            A sensitivity coefficient method for the reconstruction of
            electrical impedance tomograms.
            Clinical Physics and Physiological Measurement,
            10(3), 275–281. doi:10.1088/0143-0815/10/3/008

        The input (dv) and output (ds) is log-normalized
        """
        if normalize:
            dv = np.log(np.abs(v1)/np.abs(v0)) * np.sign(v0)
        else:
            dv = (v1 - v0)
        # s_r = J^Tv_r
        ds = -np.dot(self.J.conj().T, dv)
        return np.exp(ds) - 1.0

    def gn(self, v, x0=None, maxiter=1, gtol=1e-4, p=None, lamb=None,
           lamb_decay=1.0, lamb_min=0, method='kotre', verbose=False):
        """
        Gaussian Newton Static Solver
        You can use a different p, lamb other than the default ones in setup

        Parameters
        ----------
        v: NDArray
            boundary measurement
        x0: NDArray, optional
            initial guess
        maxiter: int, optional
            number of maximum iterations
        p, lamb: float
            JAC parameters (can be overridden)
        lamb_decay: float
            decay of lamb0, i.e., lamb0 = lamb0 * lamb_decay of each iteration
        lamb_min: float
            minimal value of lamb
        method: str, optional
            'kotre' or 'lm'
        verbose: bool, optional
            print debug information
        xp: NDArray, optional
            prior information

        Returns
        -------
        sigma: NDArray
            Complex-valued conductivities

        Note
        ----
        Gauss-Newton Iterative solver,
            x1 = x0 - (J^TJ + lamb*R)^(-1) * r0
        where:
            R = diag(J^TJ)**p
            r0 (residual) = real_measure - forward_v
        """
        from sklearn.metrics import r2_score

        if x0 is None:
            x0 = self.perm
        if p is None:
            p = self.params['p']
        if lamb is None:
            lamb = self.params['lamb']
        if method is None:
            method = self.params['method']
        

        # convergence test
        x0_norm = np.linalg.norm(x0)
        convergence = []
        r2i = []

        xp = self.params['xp']
        for i in range(maxiter):

            # forward solver
            fs = self.fwd.solve_eit(self.ex_mat, step=self.step,
                                    perm=x0, parser=self.parser)
            # Residual
            r0 = v - fs.v
            r1 = x0-xp
            jac = fs.jac

            # Damped Gaussian-Newton
            h_mat, r_mat = hr_matrix(jac, p, lamb, method)

            # update
            d_k = np.dot(h_mat, np.dot(jac.transpose(), r0) + lamb*np.dot(r_mat, r1))
            x0 = x0 - d_k

            # convergence test
            c = np.linalg.norm(d_k) / x0_norm
            r2 = r2_score(v, fs.v)
            convergence.append(c)
            r2i.append(r2)

            if c < gtol:
                break

            if verbose:
                print('iter = %d, lamb = %f, gtol = %f' % (i, lamb, c))

            # update regularization parameter
            # TODO: support user defined decreasing order of lambda series
            lamb *= lamb_decay
            if lamb < lamb_min:
                lamb = lamb_min

        self.tol = {'convergence': convergence, 'r2': r2i}

        return x0

    def project(self, ds):
        """ project ds using spatial difference filter (deprecated)

        Parameters
        ----------
        ds: NDArray
            delta sigma (conductivities)

        Returns
        -------
        NDArray
        """
        d_mat = sar(self.tri)
        return np.dot(d_mat, ds)


def h_matrix(jac, p, lamb, method='kotre', W=None):
    """
    JAC method of dynamic EIT solver:
        H = (J.T*J + lamb*R)^(-1) * J.T

    Parameters
    ----------
    jac: NDArray
        Jacobian
    p, lamb: float
        regularization parameters
    method: str, optional
        regularization method

    Returns
    -------
    H: NDArray
        pseudo-inverse matrix of JAC
    """
    if W is None:
        j_w_j = np.dot(jac.transpose(), jac)
    else:
        j_w_j = multi_dot([jac.transpose(), W, jac])

    if method == 'kotre':
        # see adler-dai-lionheart-2007
        # p=0   : noise distribute on the boundary ('dgn')
        # p=0.5 : noise distribute on the middle
        # p=1   : noise distribute on the center ('lm')
        r_mat = np.diag(np.diag(j_w_j))**p
    elif method == 'lm':
        # Marquardt–Levenberg, 'lm' for short
        # or can be called NOSER, DLS
        r_mat = np.diag(np.diag(j_w_j))
    else:
        # Damped Gauss Newton, 'dgn' for short
        r_mat = np.eye(jac.shape[1])

    # build H
    h_mat = np.dot(la.inv(j_w_j + lamb*r_mat), jac.transpose())
    return h_mat


# def h_matrix(jac, p, lamb, method='kotre'):
#     """
#     JAC method of dynamic EIT solver:
#         H = (J.T*J + lamb*R)^(-1) * J.T
# 
#     Parameters
#     ----------
#     jac: NDArray
#         Jacobian
#     p, lamb: float
#         regularization parameters
#     method: str, optional
#         regularization method
# 
#     Returns
#     -------
#     H: NDArray
#         pseudo-inverse matrix of JAC
#     """
#     j_w_j = np.dot(jac.transpose(), jac)
#     if method == 'kotre':
#         # see adler-dai-lionheart-2007
#         # p=0   : noise distribute on the boundary ('dgn')
#         # p=0.5 : noise distribute on the middle
#         # p=1   : noise distribute on the center ('lm')
#         r_mat = np.diag(np.diag(j_w_j))**p
#     elif method == 'lm':
#         # Marquardt–Levenberg, 'lm' for short
#         # or can be called NOSER, DLS
#         r_mat = np.diag(np.diag(j_w_j))
#     else:
#         # Damped Gauss Newton, 'dgn' for short
#         r_mat = np.eye(jac.shape[1])
# 
#     # build H
#     h_mat = np.dot(la.inv(j_w_j + lamb*r_mat), jac.transpose())
#     return h_mat


def hr_matrix(jac, p, lamb, method='kotre'):
    """
    JAC method of dynamic EIT solver:
        H = (J.T*J + lamb*R)^(-1) * J.T

    Parameters
    ----------
    jac: NDArray
        Jacobian
    p, lamb: float
        regularization parameters
    method: str, optional
        regularization method

    Returns
    -------
    H: NDArray
        pseudo-inverse matrix of JAC
    """
    j_w_j = np.dot(jac.transpose(), jac)
    if method == 'kotre':
        # see adler-dai-lionheart-2007
        # p=0   : noise distribute on the boundary ('dgn')
        # p=0.5 : noise distribute on the middle
        # p=1   : noise distribute on the center ('lm')
        r_mat = np.diag(np.diag(j_w_j))**p
    elif method == 'lm':
        # Marquardt–Levenberg, 'lm' for short
        # or can be called NOSER, DLS
        r_mat = np.diag(np.diag(j_w_j))
    else:
        # Damped Gauss Newton, 'dgn' for short
        r_mat = np.eye(jac.shape[1])

    # build H
    h_mat = la.inv(j_w_j + lamb*r_mat)
    return h_mat, r_mat


def qr_matrix(jac, p, lamb, W, Wx, method='kotre'):
    """
    JAC method of dynamic EIT solver:
        Q = (J.T*W*J + lamb*Wx)^(-1)

    Parameters
    ----------
    jac: NDArray
        Jacobian
    p, lamb: float
        regularization parameters
    method: str, optional
        regularization method

    Returns
    -------
    H: NDArray
        pseudo-inverse matrix of JAC
    """
    j_w_j = multi_dot([jac.transpose(), W, jac])

    if method == 'kotre':
        # see adler-dai-lionheart-2007
        # p=0   : noise distribute on the boundary ('dgn')
        # p=0.5 : noise distribute on the middle
        # p=1   : noise distribute on the center ('lm')
        r_mat = np.diag(np.diag(j_w_j)) ** p
    elif method == 'lm':
        # Marquardt–Levenberg, 'lm' for short
        # or can be called NOSER, DLS
        r_mat = np.diag(np.diag(j_w_j))
    else:
        # Damped Gauss Newton, 'dgn' for short
        r_mat = np.eye(jac.shape[1])

    # build Q
    q_mat = la.inv(j_w_j + lamb*np.dot(r_mat, Wx))
    return q_mat, r_mat


def sar(el2no):
    """
    extract spatial difference matrix on the neighbors of each element
    in 2D fem using triangular mesh.

    Parameters
    ----------
    el2no: NDArray
        triangle structures

    Returns
    -------
    D: NDArray
        SAR matrix
    """
    ne = el2no.shape[0]
    d_mat = np.eye(ne)
    for i in range(ne):
        ei = el2no[i, :]
        #
        i0 = np.argwhere(el2no == ei[0])[:, 0]
        i1 = np.argwhere(el2no == ei[1])[:, 0]
        i2 = np.argwhere(el2no == ei[2])[:, 0]
        idx = np.unique(np.hstack([i0, i1, i2]))
        # build row-i
        for j in idx:
            d_mat[i, j] = -1
        nn = idx.size - 1
        d_mat[i, i] = nn
    return d_mat