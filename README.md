# Synchronous Observer Design for Inertial Navigation

This code is an implementation of a synchronous observer design that can be used in an inertial navigation system.
It is based on the scientific article

```latex
@article{van2023synchronous,
  title={Synchronous observer design for inertial navigation systems with almost-global convergence},
  author={van Goor, Pieter and Hamel, Tarek and Mahony, Robert},
  journal={arXiv preprint arXiv:2311.02234},
  year={2023}
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