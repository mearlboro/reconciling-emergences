# Reconciling emergences: Identifying causal emergence in multivariate data

Python module for the computation of quantities for the information-theoretic
theory of emergence. See the [original Matlab repository](https://github.com/pmediano/ReconcilingEmergences) for more information.

Rosas*, Mediano*, et al. (2020). Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. PLoS Computational Biology, 16(12): e1008289. DOI: 10.1371/journal.pcbi.1008289


## Usage example
Before running any emergence calculation you must start the JVM, and functions
are provided to do so in `emergence.utils.jvm` see example below.

```python
from emergence.utils.jvm import JVM
from emergence.micalc import MutualInfo
from emergence.calc.jidt import JidtCalc

X = np.random.randint(2, size = (1000, 2))
M = np.array([ np.logical_xor(*x) for x in X ], dtype = int)
M[2:] += M[:-2]

est = MutualInfo.get('Discrete')
calc = JidtCalc(X, M, est, pointwise = False, dt = 2)
psi0 = calc.psi(q = 0)
psi1 = calc.psi(q = 1)
```

In the above example, `psi0` should be smaller than `psi1`, and any other `dt`
value should yield a considerably smaller `psi`.

## Development
### Running the code

The emergence calculator in `emergence/calc/jidt.py` makes use of [JIDT](https://github.com/jlizier/jidt)
to estimate mutual information. This is managed using `jpype`. A pre-compiled
of JIDT is available as `infodynamics.jar` in `emergence/utils`. The software
should run bug-free after installing `requirements.txt`.


### Adding new calculators
New emergence calculators can be implemented using the template below.
Most of the time no changes need be done to `psi()`, `gamma()` or `delta()`,
only the constructor.

```python
class NewCalc(EmergenceCalc):
    def __init__(self,
            X, V, mutualInfo,
            pointwise: bool = False,
            dt: int = 1,
            filename: str = ''
        ) -> None:
        super().__init__()

    def psi(self, q: int = 0) -> Union[float, List[float]]:
        pass

    def gamma(self) -> Union[float, List[float]]:
        pass

    def delta(self) -> Union[float, List[float]]:
        pass
```
