# Derivatives Pricing Library

A modular Python library for pricing financial derivatives,
built with a focus on clean architecture, numerical precision,
and extensibility.

## Architecture

The library is organized into five layers with strict separation
of concerns and a single public interface:
```
Contracts  →  payoff specification
Market     →  yield curves, discount factors
Models     →  stochastic dynamics under the pricing measure
Engines    →  numerical algorithms
Pricers    →  orchestration layer
```

Each layer depends only on the abstractions defined by the layers
below it, never on concrete implementations. This makes the library
extensible by design — new models, engines, and instruments can be
added without touching existing code.

The single public interface is `api.py`. All internal architecture
is encapsulated behind it.

## Current Functionality

### Binomial Pricing for European Options

Prices European call and put options using a recombining binomial
tree with backward induction under the risk-neutral measure.
The implementation is fully vectorized using NumPy — no Python
loops over tree nodes.
```python
import hesperides.api as hapi

price = hapi.get_price_binomial_european(
    St=100.0,   # spot price
    K=110.0,    # strike
    T=4,        # time steps to maturity
    R=0.05,     # risk-free rate per period
    u=1.25,     # up factor
    d=0.85,     # down factor
    call=True   # True for call, False for put
)
```

The implementation satisfies standard no-arbitrage properties:
put-call parity and option price bounds are verified against
known analytical results.

## Roadmap

- Black-Scholes analytical pricing
- Monte Carlo simulation engine with reproducible seeds
- PDE methods for derivative pricing
- Term structure models for interest rates
- Multiple discount curve implementations (flat, bootstrapped,
  interpolated)
- American option pricing

## Tech Stack

- Python 3.13
- NumPy (fully vectorized)
