# README #

First public release README file (v0.2, May 2025).

Here is a small guide to start experimenting with the **Python Peer-to-Peer Energy Sharing Optimization Library** (PyPESOL).

The software is shared under MIT license.

> **Disclaimer** The code is highly experimental and has only been tested on Ubuntu and MacOS. *Use with care!*

## Re-use & cite ###

If this software is used in research work, please cite the following publication:
- *Duvignau, Romaric, Vincenzo Gulisano, and Marina Papatriantafilou. "PyPESOL: The Python P2P Energy Sharing Optimization Library." Proceedings of the Sixteenth ACM International Conference on Future Energy Systems (ACM e-energy '25). 2025. 1014-1015.* https://dl.acm.org/doi/10.1145/3679240.3734691


The library implements energy and optimization models as well as matching methods from the following publications:
- Duvignau, R., Gulisano, V., Papatriantafilou, M., & Klasing, R. (2024). Geographical Peer Matching for P2P Energy Sharing. IEEE Access.
- Duvignau, R., & Klasing, R. (2023). Greediness is not always a vice: Efficient Discovery Algorithms for Assignment Problems. Procedia Computer Science, 223, 43-52.
- Duvignau, R., Gulisano, V., & Papatriantafilou, M. (2023, January). Cost-optimization for win-win P2P energy systems. In 2023 IEEE Power & Energy Society Innovative Smart Grid Technologies Conference (ISGT) (pp. 1-5). IEEE.
- Duvignau, R., Gulisano, V., & Papatriantafilou, M. (2022, April). Efficient and scalable geographical peer matching for p2p energy sharing communities. In Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing (pp. 187-190).
- Duvignau, R., Heinisch, V., Göransson, L., Gulisano, V., & Papatriantafilou, M. (2021). Benefits of small-size communities for continuous cost-optimization in peer-to-peer energy sharing. Applied Energy, 301, 117402.
- Duvignau, R., Heinisch, V., Göransson, L., Gulisano, V., & Papatriantafilou, M. (2020, June). Small-scale communities are sufficient for cost-and data-efficient peer-to-peer energy sharing. In Proceedings of the Eleventh ACM International Conference on Future Energy Systems (pp. 35-46).


## For a New Fresh Install ###

1. Clone the source code.

2. Packages to install before running the code: (required) pyomo, (optional for some sub-components) numpy, scipy, pandas, prophet (**as of prophet v1.1, requires Python3.7**), plotly eg using the current Python3.7 installation:


```
pip3 install numpy scipy pyomo pandas plotly prophet
```

or complete install via conda and a virtual environment for *Python 3.7*, including two optimization solvers:

```
conda update anaconda
conda create -n py37 python=3.7
conda activate py37
conda install numpy scipy pandas plotly prophet redis
conda install -c conda-forge pyomo
conda install -c conda-forge ipopt glpk
```


3. Install a LP solver program. The default one that is configured in the code is 'cbc' solver. 


### Ubuntu

```
sudo apt-get install coinor-cbc
```

### MacOS

```
brew install cbc
```

## Getting started

Follow the [jupyter notebook tutorial](getting_started.ipynb)

## Input -- Datasets ###

Inspect the following files:

* **cons.csv**

Household consumptions, each line is 1 hour of consumption.

That is: h1, h2, ..., hn consumption for hour1, then a newline and
the same for hour3 and so on.

The size is then N*365*24 numbers, N values per line, where N is number of households.
(default unit kWh)

* **price.csv**

Yearly hourly price, so 365*24 numbers, 1 value per line.
(default unit €/kWh)

* **sun.csv**

Solar profile, so 365*24 numbers, 1 value per line.
(default unit kWh/kWp)

* **pv.csv**

File with PV capacities.

* **battery.csv**

File with battery capacities.
