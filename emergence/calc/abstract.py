"""
Abstract class for calculating the quantities related to the theory of
causal emergence, as described in:

Rosas FE*, Mediano PAM*, Jensen HJ, Seth AK, Barrett AB, Carhart-Harris RL, et
al. (2020) Reconciling emergences: An information-theoretic approach to
identify causal emergence in multivariate data. PLoS Comput Biol 16(12):
e1008289.
"""
from abc import ABCMeta, abstractmethod
import itertools as it
import logging
import math
import numpy as np
import pickle
from typing import Callable, List, Union

import emergence.utils.log

class BaseCalc(metaclass = ABCMeta):
    """
    Computes quanities related to causal emergence using a given MutualInfo calculator
    function on time series X and V
    """
    @abstractmethod
    def __init__(self,
            X: np.ndarray, V: np.ndarray,
            mutualInfo: Callable[[np.ndarray, np.ndarray, bool, int], float],
            pointwise: bool = False, dt: int = 1, filename: str = ''
        ) -> None:
        """
        Initialise class with the time series corresponding to the parts Xs and
        the whole V, and set any properties for the computation. By default, works
        with multivariate systems, so if any txn 2D array of t states of n scalar
        variable is passed, reshape the array to (t, n, 1) first.

        Params
        ------
        X
            system micro variables of shape (t, n, d) for n d-dimensional time
            series corresponding to the 'parts' in the system
        V
            candidate emergence feature: d-dimensional system macro variable of
            shape (t, d)
        mutualInfo
            mutual information function to use from MutualInfo class
        pointwise
            whether to use pointwise (p log p) or Shannon (sum p log p) MI
        dt
            number of time steps in the future to predict
        filename
            if set, save the object to a file
        """
        logging.info(f"Initialise Emergence Calculator using {'pointwise' if pointwise else 'Shannon'} mutual information with t'=t+{dt}")
        logging.info(f"  and MI estimator {mutualInfo.__name__}")

        if len(X.shape) < 3:
            X = np.atleast_3d(X)
        if len(V.shape) < 2:
            V = V[:, np.newaxis]

        (t, n, d) = X.shape
        self.n = n
        self.V = V
        self.X = [ X[:, i] for i in range(n) ]
        logging.info(f"  for {n} {d}-dimensional variables and {V.shape[1]}-dimensional macroscopic feature")
        logging.info(f"  as time series of length {t}")

        logging.info(f"Computing mutual informations: MI(V[t], V[t+{dt}])")
        self.vmiCalc = mutualInfo(V,  V,  pointwise, dt)

        logging.info(f"Computing mutual informations: MI(Xi[t], V[t+{dt}])")
        self.xvmiCalcs = dict()
        for i in range(n):
            self.xvmiCalcs[i] = mutualInfo(self.X[i], V, pointwise, dt)

        logging.info(f"Computing mutual informations: MI(V[t], Xi[t+{dt}])")
        self.vxmiCalcs = dict()
        for i in range(n):
            self.vxmiCalcs[i] = mutualInfo(V, self.X[i], pointwise, dt)

        logging.info(f"Computing mutual informations: MI(Xi[t], Xj[t+{dt}])")
        self.xmiCalcs = dict()
        for i in range(n):
            for j in range(n):
                self.xmiCalcs[(i, j)] = mutualInfo(self.X[i], self.X[j], pointwise, dt)

        if filename:
            logging.info(f"Dumping EmergenceCalc object with all pairwise MI to {name}_calc.pkl")
            with open(f"{filename}_calc.pkl", 'wb') as f:
                f.dump(self, f, pickle.HIGHEST_PROTOCOL)

        logging.info('Done.')


    @classmethod
    def correction_coef(cls, n: int, q: int, r: int) -> int:
        """
        Compute coefficient for the double-counting redundancy correction

        Params
        ------
        n
            system size i.e. number of microscopic features (sources)
        q
            order of correction, strictly smaller than n
        r
            size of the set of sources being considered in the redundancy
            calculation
        """
        if q > n - 1:
            err = f"Order of correction q={q} must be strictly smaller than number of sources n={n}"
            logging.error(err)
            raise ValueError(err)
        if r > n or r < n - q + 1:
            err = f"Atom set size r={r} must be strictly between {n - q + 1} and {n}"
            logging.error(err)
            raise ValueError(err)

        if r == n - q + 1:
            return n - q
        else:
            return r - 1 - sum( cls.correction_coef(n, q, s) * math.comb(r, s)
                                for s in range(n - q + 1, r) )


    def _intersection_info(self, ixs: List[int]) -> float:
        """
        Compute intersection information of all terms with indices in the given
        list, i.e. the minimum mutual info

        Params
        ------
        ixs
            list of indices representing information atoms
        """
        infos = [ self.xvmiCalcs[i] for i in ixs ]
        return min(infos)


    def _lattice_expansion(self, q: int = 0) -> List[int]:
        """
        Expand PID latice for qth order correction.
        Given a system of size n (i.e. n sources), corrections are supported
        up to order q=n-1. The correction consists of adding (and if needed,
        subtracting) the intersection information from expanding the lattice.
        The intersection (redundant) info is defined as minimal mutual info (MMI)

        - The 1st order correction adds the MMI of all n redundant atoms to the
        uncorrected Psi.
        - The 2nd order correction adds a sum of MMIs over sets of n-1 redundant
        atoms and subtracts the MMI of all n redundant atoms.
        - The qth order correction adds sums over MMI over sets of n-q+1 redundant
        atoms and subtracts the MMI over sets of n-q+2 atoms, and so on
        """
        n = self.n

        corr = 0

        for r in range(n - q + 1, n + 1):
            logging.info(f"Computing correction for atom sets of size {r}")
            atom_sets = list(it.combinations(range(n), r))
            coef = self.correction_coef(n, q, r)
            logging.info(f"with {len(atom_sets)} sets with coefficient {coef}")

            corr += sum(
                coef * self._intersection_info(ixs) for ixs in atom_sets )

        return corr


    @abstractmethod
    def psi(self, q: int = 0) -> Union[float, List[float]]:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Psi = Synergy - Redundancy + Correction

        where:
            Synergy     MI(V(t); V(t'))
            Redundancy  sum_i MI(X_i(t); V(t'))
            Correction  lattice_expansion(MI(X(t);V(t')), q)

        where  t' - t = self.dt

        Params
        ------
        q
            compute lattice correction of order q, i.e. add back the partial
            mutual information corresponding to the q lowest levels of the PID
            lattice

        Returns
        ------
        psi
            the synergy-redundancy index with redundancy correction of order q
        """
        msg = "Computing Psi"
        if q:
            msg += f" using redundancy correction of order {q}"
        logging.info(msg)
        if q > self.n - 1:
            err = f"Order of correction q={q} must be strictly smaller than number of sources n={self.n}"
            logging.error(err)
            raise ValueError(err)


    @abstractmethod
    def gamma(self) -> Union[float, List[float]]:
        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Gamma = max_j I(V(t); X_j(t'))

        where  t' - t = self.dt
        """
        pass

    @abstractmethod
    def delta(self) -> Union[float, List[float]]:

        """
        Use MI quantities computed in the intialiser to derive practical criterion
        for emergence.

            Delta = max_j (I(V(t);X_j(t')) - sum_i I(X_i(t); X_j(t'))

        where  t' - t = self.dt
        """
        pass

