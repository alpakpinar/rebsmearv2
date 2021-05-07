# Rebalance and smear

Flexible implementation of the rebalance and smear method in python + RooFit. This package also introduces the ability to send jobs to condor.

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

## Run in HTCondor

Using the `rebexec` command line tool, one can submit jobs to HTcondor servers:

```python
rebexec submit --dataset 'QCD_HT.*2017' --name 'my_rebsmear_submission'
```

There are some testing options for use: Use `--dry` to run the script but do not make the actual submission. Use `--test` to run 1 file per dataset.

## Run locally

`rebexec` also suppors running the jobs locally:

```python
rebexec run --dataset 'QCD_HT.*2017' --name 'my_rebsmear_submission'
```
