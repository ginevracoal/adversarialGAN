#### Master thesis 2020 - Data Science & Scientific Computing
# Adversarial Learning of Robust and Safe Controllers for Cyber-Physical Systems 

## Abstract
Data-driven models are bringing huge advantages in terms of performance and effectiveness in many sectors.
In the field of Cyber-Physical Systems, although, they are not widespread since their opaqueness and lacks formal proofs pose a serious threat to their employment.

We propose and analyze a novel approach in creating robust and safe controllers drawing from the literature of the adversarial learning.

Our method trains in an adversarial way two Neural Networks to reach a twofold goal:
obtain one network that is able to generate difficult configurations of the environment and another that is able to overcome them in a safe e robust way.
The aim is to create a formally verified controller and, at the same time, to give insights on the most demanding corner cases of a given model.

The approach is promising and worthy of further investigation.

## Project structure
The project has been developed in PyTorch and still needs to undergo a heavy refactoring.
The main directories are:
- `doc`, contains the Latex source code of the whole thesis
- `misc`, contains the thesis and the presentation slides in PDF
- `src`, the actual source code of the architecture developed

### Source code
The project provides 3 main modules to create the desired experimental setup.
- `architecture.py` provides the main structures of the whole concept (NNs, training and testing procedure)
- `diffquantitative.py` provides the logic to write, parse and check STL formulae
- `misc.py` groups some minor helper functions

Each experimental setup is composed of:
- the _**model**_ of the world that includes the definition of the _attacker_ and _defender_ and the differential equation that describes their evolution over time (`model_*.py`)
- the _**training setup**_ that defines the configuration of the architecture employed in the experiment (`train_*.py`)
- _(optional)_ the _**testing setup**_ that evaluates the trained model in controlled scenarios and stores the experimental data (`tester_*.py`)
- _(optional)_ the _**plotting**_ scripts that read the experimental data and produce the required visualizations (`plotter_*.py`)

**Note**: the tests performed by the class `architecture.Tester` can only measure the performance of the training process, they are not useful to assess the experimental setup.

## Quick start
Once the repository has been cloned, create a python3 _virtual environment_ and install the specified requirements.
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r src/requirements.txt
```
Once the environment is ready, it's possible to launch the training of a specific model:
```
cd src
python train_platooning.py
```


## Disclaimer
The code, despite its robustness, is far from being production-ready: it needs to be refactored and, in some parts, it requires a redesign to allow a faster prototypation of experimental models and setups. Therefore it should be considered as a Proof Of Concept.

## Licence
Creative Commons 4.0 CC-BY
[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)
