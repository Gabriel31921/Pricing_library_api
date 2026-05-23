import numpy as np

from hesperides.contracts.options import EuropeanOption
from hesperides.market.curves import FlatDiscountCurve
from hesperides.market import static_arbitrage as _sa
from hesperides.models.binomial import BinomialModel
from hesperides.pricers.binomial_pricer import BinomialPricer


def get_price_binomial_european(
    St: float,
    K: float,
    T: int,
    R: float,
    u: float,
    d: float,
    call: bool,
) -> float:
    """
    Price a European call or put option using the binomial model.

    Parameters
    ----------
    St : float
        Spot price of the underlying at time t.
    K : float
        Strike price.
    T : int
        Number of time steps to maturity.
    R : float
        Risk-free rate (per period).
    u : float
        Up factor.
    d : float
        Down factor.
    call : bool
        If True, price a call option; if False, price a put option.

    Returns
    -------
    float
        Option price at time t.
    """
    contract = EuropeanOption(K, call)
    curve = FlatDiscountCurve(R)
    model = BinomialModel(u, d, St)
    pricer = BinomialPricer(contract, curve, model, T)

    Pi = pricer.pricing()

    return Pi

def compute_static_arbitrage_quantity(
    surface: np.ndarray,
    strikes: np.ndarray | None = None,
    quantity: str = "vertical",
) -> np.ndarray:
    """
    Carr–Madan static-arbitrage spread grids on a call surface C_{i,j}.

    Parameters
    ----------
    surface : ndarray, shape (nK, nT)
        Call prices by strike (row) and expiry (column).
    strikes : ndarray, shape (nK,) or None
        Strictly increasing strikes. Required for ``quantity`` ``'vertical'`` and
        ``'butterfly'``; ignored for ``'calendar'``.
    quantity : {'vertical', 'butterfly', 'calendar'}, optional
        Which spread grid to return.

    Returns
    -------
    ndarray
        * ``'vertical'``: normalized vertical call spreads, shape (nK-1, nT).
          For K_0 < ... < K_{nK-1} and i = 1, ..., nK-1,

          Q_{i,j} = (C_{i-1,j} - C_{i,j}) / (K_i - K_{i-1}).

        * ``'butterfly'``: interior butterfly values (at least three strikes),
          shape (nK-2, nT), matching Carr–Madan’s construction after equation (1).

        * ``'calendar'``: calendar spreads across consecutive expiries,
          shape (nK, nT-1),

          CS_{i,j} = C_{i,j+1} - C_{i,j}.
    """

    return _sa.compute(surface = surface, strikes = strikes, quantity = quantity) #type: ignore
