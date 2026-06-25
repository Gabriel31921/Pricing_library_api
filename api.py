import numpy as np

from hesperides.contracts.options import EuropeanOption
from hesperides.market.curves import FlatDiscountCurve
from hesperides.market import static_arbitrage as _sa
from hesperides.models.binomial import BinomialModel
from hesperides.pricers.binomial_pricer import BinomialPricer
from hesperides.models.black_scholes import BlackScholesModel
from hesperides.pricers.bs_pricer import BSPricer
from hesperides.contracts.options import GeometricAsianOption
from hesperides.market.bumps import default_bump


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

def get_price_bs_european(
    St: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool,
    engine: str = "analytical",
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Price a European option (call or put) under Black-Scholes.

    Parameters
    ----------
    St : float
        Spot price of the underlying at valuation date.
    K : float
        Strike.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Black-Scholes volatility (annualized).
    call : bool
        True for call, False for put.
    engine : {"analytical", "mc"}, optional
        Pricing engine. Default "analytical".
    n_paths : int or None, optional
        Number of Monte Carlo paths. Required if engine="mc".
    seed : int or None, optional
        Seed for reproducible Monte Carlo.

    Returns
    -------
    float
        Option price at valuation date.
    """
    if engine not in ("analytical", "mc"):
        raise ValueError(f"engine must be 'analytical' or 'mc', got {engine!r}")
    if engine == "mc" and n_paths is None:
        raise ValueError("n_paths is required when engine='mc'")

    contract = EuropeanOption(K, call)
    model = BlackScholesModel(St, r, sigma)
    pricer = BSPricer(contract, model, T)

    if engine == "analytical":
        return pricer.price_analytical()
    else:
        return pricer.price_mc(n_paths=n_paths, n_steps=1, seed=seed)  # type: ignore[arg-type]

def get_price_bs_european_dividend(
    St: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool,
    q: float = 0.0,
    engine: str = "analytical",
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Price a European option on a stock paying a continuous dividend yield q.

    Risk-neutral dynamics dS = (r - q) S dt + sigma S dW (Module 9);
    discounting still uses r. With q = 0 this recovers Assignment 3.

    Parameters
    ----------
    St, K, T, r, sigma, call, engine, n_paths, seed
        As in ``get_price_bs_european`` (Assignment 3).
    q : float, optional
        Continuous dividend yield. Default 0.0 (recovers Assignment 3).

    Returns
    -------
    float
        Option price at valuation date.
    """
    if engine not in ("analytical", "mc"):
        raise ValueError(f"engine must be 'analytical' or 'mc', got {engine!r}")
    if engine == "mc" and n_paths is None:
        raise ValueError("n_paths is required when engine='mc'")
    if St <= 0:
        raise ValueError(f"St must be > 0, got {St}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")

    contract = EuropeanOption(K, call)
    model = BlackScholesModel(St, r, sigma, q)
    pricer = BSPricer(contract, model, T)

    if engine == "analytical":
        return pricer.price_analytical()
    return pricer.price_mc(n_paths=n_paths, n_steps=1, seed=seed)  # type: ignore[arg-type]

def get_price_fx_option(
    St: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    call: bool,
    engine: str = "analytical",
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Price a European FX option under the Garman-Kohlhagen model.

    St is the FX spot (domestic per unit foreign); r_d and r_f are the
    domestic and foreign continuously compounded rates; the price is returned
    in domestic currency. Internally this is the cost-of-carry model with
    q = r_f and r = r_d.

    Returns
    -------
    float
        Option price at valuation date, in domestic currency.
    """
    if engine not in ("analytical", "mc"):
        raise ValueError(f"engine must be 'analytical' or 'mc', got {engine!r}")
    if engine == "mc" and n_paths is None:
        raise ValueError("n_paths is required when engine='mc'")
    if St <= 0:
        raise ValueError(f"St must be > 0, got {St}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")

    # Garman-Kohlhagen = cost of carry with q = r_f, discounting at r_d.
    contract = EuropeanOption(K, call)
    model = BlackScholesModel(St, r_d, sigma, r_f)
    pricer = BSPricer(contract, model, T)

    if engine == "analytical":
        return pricer.price_analytical()
    return pricer.price_mc(n_paths=n_paths, n_steps=1, seed=seed)  # type: ignore[arg-type]

def get_price_future_option(
    F0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool,
    engine: str = "analytical",
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Price a European option on a future under the Black-76 model.

    F0 is the current future price. Under Q the future is driftless,
    dF = sigma F dW, i.e. the cost-of-carry model with zero carry (q = r);
    discounting uses r. With these inputs the price is the Black-76 value
    e^{-rT} (F0 N(d_+) - K N(d_-)) for a call. No separate "futures model":
    reuse the same Black-Scholes model with q = r.

    Returns
    -------
    float
        Option price at valuation date.
    """
    if engine not in ("analytical", "mc"):
        raise ValueError(f"engine must be 'analytical' or 'mc', got {engine!r}")
    if engine == "mc" and n_paths is None:
        raise ValueError("n_paths is required when engine='mc'")
    if F0 <= 0:
        raise ValueError(f"F0 must be > 0, got {F0}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")

    # Black-76 = cost of carry with zero carry (q = r); discounting at r.
    contract = EuropeanOption(K, call)
    model = BlackScholesModel(F0, r, sigma, r)
    pricer = BSPricer(contract, model, T)

    if engine == "analytical":
        return pricer.price_analytical()
    return pricer.price_mc(n_paths=n_paths, n_steps=1, seed=seed)  # type: ignore[arg-type]

def get_price_bs_geometric_asian(
    St: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool,
    engine: str = "analytical",
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Price a geometric Asian option (call or put) under Black-Scholes.

    Parameters
    ----------
    St, K, T, r, sigma, call, engine, n_paths, seed
        As in ``get_price_bs_european``.
    n_steps : int or None, optional
        Number of time steps in the Monte Carlo grid. Required if engine="mc".

    Returns
    -------
    float
        Option price at valuation date.
    """
    if engine not in ("analytical", "mc"):
        raise ValueError(f"engine must be 'analytical' or 'mc', got {engine!r}")
    if engine == "mc":
        if n_paths is None:
            raise ValueError("n_paths is required when engine='mc'")
        if n_steps is None:
            raise ValueError("n_steps is required for geometric Asian when engine='mc'")

    contract = GeometricAsianOption(K, call)
    model = BlackScholesModel(St, r, sigma)
    pricer = BSPricer(contract, model, T)

    if engine == "analytical":
        return pricer.price_analytical()
    else:
        return pricer.price_mc(n_paths=n_paths, n_steps=n_steps, seed=seed)  # type: ignore[arg-type]

def get_greek_bs_european(
    St: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool,
    greek: str,
    engine: str = "analytical",
    greek_engine: str = "analytical",
    fd_scheme: str = "central",
    h: float | None = None,
    n_paths: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Compute a Greek of a European call/put under Black–Scholes.

    Two orthogonal axes drive the computation:

    * ``engine`` selects the underlying *pricing* engine: ``"analytical"``
      (closed-form Black–Scholes price) or ``"mc"`` (Monte Carlo).
    * ``greek_engine`` selects the *Greek* method: ``"analytical"``
      (closed-form derivative of the BS price; no pricing engine
      involved) or ``"fd"`` (finite-difference bump-and-reprice of the
      configured pricing engine).

    Parameters
    ----------
    St : float
        Spot price of the underlying at valuation date.
    K : float
        Strike.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Black–Scholes volatility (annualized).
    call : bool
        True for call, False for put.
    greek : {"delta", "gamma", "vega", "rho"}
        Which sensitivity to return.
    engine : {"analytical", "mc"}, optional
        Pricing engine. Used only when ``greek_engine="fd"``. With
        ``engine="analytical"`` the FD engine wraps the closed-form
        pricer (sanity test, no Monte Carlo noise). With ``engine="mc"``
        it wraps the Monte Carlo pricer (production scenario; requires
        both ``n_paths`` and a fixed ``seed`` so bumped repricings share
        the same random numbers). Default
        ``"analytical"``.
    greek_engine : {"analytical", "fd"}, optional
        Greek computation method. ``"analytical"`` returns the
        closed-form Black-Scholes Greek directly; after selector
        validation, ``fd_scheme``, ``h``, ``n_paths`` and ``seed`` are
        ignored. ``"fd"`` applies finite-difference bump-and-reprice on
        top of the chosen ``engine``. Default ``"analytical"``.
    fd_scheme : {"forward", "central"}, optional
        Finite-difference scheme for first-order Greeks. Only used when
        ``greek_engine="fd"``. Default ``"central"``. Gamma always uses
        the central second-difference regardless of this argument (a
        second derivative needs evaluations at +h, 0 and -h).
    h : float or None, optional
        Bump size, applied additively in the natural units of whichever
        parameter is being bumped (St for delta/gamma, sigma for vega,
        r for rho). If None, choose sensible per-parameter defaults;
        the default spot bump may be proportional to St. Only used when
        ``greek_engine="fd"``.
    n_paths : int or None, optional
        Number of Monte Carlo paths. Required when
        ``greek_engine="fd"`` and ``engine="mc"``; ignored otherwise.
    seed : int or None, optional
        Seed for the Monte Carlo engine. Required when
        ``greek_engine="fd"`` and ``engine="mc"`` so the bumped
        repricings use common random numbers. Ignored otherwise.

    Returns
    -------
    float
        The requested Greek.
    """
    # --- Selector validations: always run, even for greek_engine="analytical" ---
    if greek not in ("delta", "gamma", "vega", "rho"):
        raise ValueError(
            f"greek must be 'delta', 'gamma', 'vega' or 'rho', got {greek!r}"
        )
    if engine not in ("analytical", "mc"):
        raise ValueError(f"engine must be 'analytical' or 'mc', got {engine!r}")
    if greek_engine not in ("analytical", "fd"):
        raise ValueError(
            f"greek_engine must be 'analytical' or 'fd', got {greek_engine!r}"
        )

    # --- Positivity of market inputs (T == 0 is not priced) ---
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if St <= 0:
        raise ValueError(f"St must be > 0, got {St}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    contract = EuropeanOption(K, call)
    model = BlackScholesModel(St, r, sigma)
    pricer = BSPricer(contract, model, T)

    # --- Closed-form Greek: fd_scheme/h/n_paths/seed are ignored here ---
    if greek_engine == "analytical":
        return pricer.greek(greek, engine, greek_engine, fd_scheme)

    # --- greek_engine == "fd": validate FD-specific arguments ---
    if fd_scheme not in ("forward", "central"):
        raise ValueError(
            f"fd_scheme must be 'forward' or 'central', got {fd_scheme!r}"
        )
    if h is not None and h <= 0:
        raise ValueError(f"h must be > 0, got {h}")
    if engine == "mc":
        if n_paths is None or n_paths <= 0:
            raise ValueError("n_paths must be > 0 when greek_engine='fd' and engine='mc'")
        if seed is None:
            raise ValueError(
                "seed is required when greek_engine='fd' and engine='mc' "
                "(common random numbers)"
            )

    if h is None:
        h = default_bump(greek, St)

    return pricer.greek(greek, engine, greek_engine, fd_scheme, h, n_paths, seed)