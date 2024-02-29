"""
The class below is built to use the JIDT mutual information calculators, as
defined in `micalc.py`, but can support any mutual information function with
the following signature:

    X  : np.array of shape (T, N, D), micro components
    Y  : np.array of shape (T, D), macro features
    pt : boolean, use pointwise mutual information
    dt : time delay

The mutual information function must return a scalar value in order for the
computation of Psi, Gamma and Delta to be robust.
"""
import numpy as np
from typing import Callable

from emergence.calc.abstract import BaseCalc


class EmergenceCalc(BaseCalc):
    def __init__(self,
            X: np.ndarray, V: np.ndarray,
            mutualInfo: Callable[[np.ndarray, np.ndarray, bool, int], float],
            pointwise: bool = False,
            dt: int =  1, filename: str = ''
        ) -> None:
        super().__init__(X, V, mutualInfo, pointwise, dt, filename)

    def psi(self, q: int = 0) -> float:
        super().psi()
        syn  = self.vmiCalc
        red  = sum(xvmi for xvmi in self.xvmiCalcs.values())
        corr = self._lattice_expansion(q)
        return syn - red + corr

    def gamma(self) -> float:
        gamma = max(self.vxmiCalcs.values())
        return gamma


    def delta(self) -> float:
        delta = max(vx - sum(self.xmiCalcs[(i, j)] for i in range(self.n))
                    for j, vx in enumerate(self.vxmiCalcs.values()) )
        return delta

