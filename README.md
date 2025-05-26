# Synchronous Observer Design for Inertial Navigation

This code is an implementation of a synchronous observer design that can be used in an inertial navigation system.
It is based on the scientific article

```latex
@article{van2025synchronous,
  title = {Synchronous observer design for Inertial Navigation Systems with almost-global convergence},
  journal = {Automatica},
  volume = {177},
  pages = {112328},
  year = {2025},
  issn = {0005-1098},
  doi = {https://doi.org/10.1016/j.automatica.2025.112328},
  url = {https://www.sciencedirect.com/science/article/pii/S0005109825002213},
  author = {Pieter {van Goor} and Tarek Hamel and Robert Mahony},
}
```

## Requirements

The python libraries used in this code are listed below.
For quick installation of all these packages, use 

```commandline
pip install numpy matplotlib scipy pylieg pymavlink progressbar2
```

* numpy: `pip install numpy`
* matplotlib: `pip install matplotlib`
* scipy: `pip install scipy`
* pylie: `pip install pylieg`
* pymavlink: `pip install pymavlink` *Not required for simulation.*
* progressbar2: `pip install progressbar2` *Not required.*

## Usage

Simulations can be run using

```commandline
python3 auto_ins_sim.py
```

The observer can also be used on ardupilot log data using

```commandline
python3 auto_ins_real.py
```

This is set up to use the log file `striver-2023-02-18-f22.bin`, which can be downloaded at `URL`.
The log file used can be changed by adjusting the code.