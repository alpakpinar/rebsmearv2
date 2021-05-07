# Rebalance and smear

Flexible implementation of the rebalance and smear method in python + RooFit.

## Example

```python
from rebalance import Jet, RebalanceWSFactory

# Event with two arbitrary reco-level jets
jets = [
    Jet(pt=100, eta=1.3, phi=0.),
    Jet(pt=100, eta=1.3, phi=2.8),
]

# Factory creates RooWorkspace for us
# default workspace has dummy assumptions for priors, resolution functions
# -> Simple extension possible via inheritance!
rbwsfac = RebalanceWSFactory(jets)
rbwsfac.build()
ws = rbwsfac.get_ws()

# Ready to minize the negative log likelihood!
m = r.RooMinimizer(ws.function("nll"))
m.migrad()
```