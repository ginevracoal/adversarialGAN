# Adversarial Learning of Robust and Safe Controllers for Cyber-Physical Systems 

Code for paper "Adversarial Learning of Robust and Safe Controllers for Cyber-Physical Systems", Luca Bortolussi, Francesca Cairoli, Ginevra Carbone, Francesco Franchina, 2020.

## Abstract
We introduce a novel learning-based approach to synthesize safe and robust con- trollers for autonomous Cyber-Physical Systems and, at the same time, to generate challenging tests. This procedure combines formal methods for model verification with Generative Adversarial Networks. The method learns two Neural Networks: the first one aims at generating troubling scenarios for the controller, while the second one aims at enforcing the safety constraints. We test the proposed method on a variety of case studies.

## Project structure

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
pip install -r requirements.txt
```
Once the environment is ready, it's possible to launch the training of a specific model:
```
cd src
python train_platooning.py
```


## Licence
Creative Commons 4.0 CC-BY
[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)
