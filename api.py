from hesperides.contracts.options import EuropeanOption
from hesperides.market.curves import FlatDiscountCurve
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