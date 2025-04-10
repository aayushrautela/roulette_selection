# EARIN 2024 Lab3: Evolutionary and genetic algorithms

## Repo setup

* Create and activate python environment. You can use either [venv](https://docs.python.org/3/library/venv.html)
  or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).
* Install the required packages:

```bash
pip install -r requirements.txt
```

How to Run

Install dependencies:

pip install -r requirements.txt

Run experiments:

python main.py

Edit main.py to switch between experiments or run them all in sequence.

Implemented GA Features

Selection: Roulette Wheel

Crossover: Random interpolation

Mutation: Gaussian perturbation

Fitness: Inverse of Booth function (minimization → maximization)