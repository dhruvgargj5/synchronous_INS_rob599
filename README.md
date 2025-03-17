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

The python libraries used in this code are:

* numpy: `pip3 install numpy`
* matplotlib: `pip3 install matplotlib`
* scipy: `pip3 install scipy`
* pylie: <https://github.com/pvangoor/pylie>
* pymavlink: `pip3 install pymavlink` *Not required for simulation.*
* progressbar2: `pip3 install progressbar2` *Not required.*

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