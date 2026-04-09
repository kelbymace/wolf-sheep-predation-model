# Wolf Sheep Predation Model in Python with Reinforcement Learning

This project is a Python implementation and extension of NetLogo’s classic **Wolf Sheep Predation** model. It began as a translation of the original model into Python and evolved into an experimental framework for comparing heuristic and reinforcement learning policies for sheep survival under predation and resource scarcity.

The main research direction of the project is:

> Can reinforcement learning produce sheep movement policies that outperform and appear more 'life-like' than simple hand-coded heuristics such as `avoid_wolves`, especially when grass is scarce and the sheep must balance predator avoidance with foraging?

---

## Features

- Python implementation of the Wolf-Sheep-Grass model
- Multiple sheep movement strategies, including:
  - `random`
  - `avoid_wolves`
  - `flock`
  - `rl`
- Multiple wolf strategies, including:
  - `random`
  - `seek_sheep`
- Grass growth and energy dynamics
- Animation and visualization with `matplotlib`
- Behavior cloning / imitation learning from a heuristic expert
- Policy-gradient reinforcement learning for sheep control
- Benchmarking tools for evaluating policies across scenarios

---

## Project Structure

```text
wolf-sheep-predation-model/
│
├── .venv/
├── .gitignore
├── README.md
├── policies/
│   └── *.pt
├── notebooks/
│   └── WSPRunner.ipynb
├── src/
│   └── wolf_sheep_rl/
│       ├── __init__.py
│       ├── model.py
│       ├── animal.py
│       ├── sheep.py
│       ├── wolf.py
│       ├── observations.py
│       ├── rewards.py
│       ├── policy.py
│       ├── training.py
│       ├── evaluation.py
│       └── visualization.py
```

---

## Main Ideas

### 1. Translation from NetLogo
The original NetLogo Wolf Sheep Predation model was translated into Python to create a more extensible experimental environment.

### 2. Heuristic Sheep Policies
Several hand-coded sheep behaviors were implemented, such as:
- moving randomly
- moving away from nearby wolves
- moving toward nearby sheep to form flocks

### 3. Reinforcement Learning
A policy network was trained to control sheep movement using:
- local observations of wolves, grass, and current energy
- discrete movement actions over the Moore neighborhood
- imitation learning from the `avoid_wolves` heuristic
- policy-gradient fine-tuning

### 4. Evaluation
Policies are compared systematically across scenarios such as:
- default conditions
- scarce grass
- multiple wolves
- smaller and larger maps

---

## Installation

Clone the repository and create a virtual environment:

```bash
git clone <your-repo-url>
cd wolf-sheep-predation-model
python -m venv .venv
```

Activate the environment:

### Windows PowerShell
```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using Jupyter notebooks:

```bash
pip install ipykernel jupyter
python -m ipykernel install --user --name wolf-sheep-rl --display-name "Python (wolf-sheep-rl)"
```

---

## Dependencies

Core libraries used in this project include:

- `numpy`
- `matplotlib`
- `pandas`
- `torch`
- `jupyter` / `ipykernel` for notebooks

You may also want:
- `pytest` for tests
- `seaborn` or `plotly` for additional plotting, if desired

---

## Running the Model

After setting your Python path appropriately or working from the notebook, you can import the package like this:

```python
from wolf_sheep_rl.model import WolfSheepModel
from wolf_sheep_rl.visualization import animate_model
```

Example:

```python
model = WolfSheepModel(
    width=25,
    height=25,
    initial_number_sheep=40,
    initial_number_wolves=8,
    model_version="sheep-wolves-grass",
    sheep_strategy="avoid_wolves",
    wolf_strategy="seek_sheep",
    enable_grass=True,
)
model.setup()
```

---

## Training an RL Policy

The RL pipeline uses:
1. behavior cloning from the `avoid_wolves` policy
2. policy-gradient fine-tuning

Example:

```python
from wolf_sheep_rl.training import train_policy_gradient

policy_net, episode_lengths = train_policy_gradient(
    num_episodes=2000,
    gamma=0.95,
    pretrain_with_expert=True,
    pretrain_samples=8000,
    pretrain_epochs=15,
    pretrain_lr=1e-3,
    model_kwargs={
        "width": 30,
        "height": 30,
        "initial_number_sheep": 25,
        "initial_number_wolves": 15,
        "model_version": "rl-training",
        "sheep_strategy": "rl",
        "wolf_strategy": "seek_sheep",
        "enable_grass": True,
        "sheep_sight_radius": 2,
        "wolf_sight_radius": 2,
        "grass_regrowth_time": 30,
    }
)
```

---

## Saving and Loading Policies

Trained PyTorch policies can be saved as `.pt` files:

```python
import torch
torch.save(policy_net.state_dict(), "policies/sheep_policy.pt")
```

Load later with:

```python
from wolf_sheep_rl.policy import PolicyNetwork
import torch

policy_net = PolicyNetwork(input_dim=51, hidden_dim=32, num_actions=8)
policy_net.load_state_dict(torch.load("policies/sheep_policy.pt"))
policy_net.eval()
```

---

## Benchmarking Policies

Policies can be evaluated under identical seeded scenarios to compare mean episode length and other metrics.

Typical comparisons include:
- `random`
- `avoid_wolves`
- `rl`

Metrics may include:
- mean episode length
- survival rate
- final energy
- number of close wolf encounters

---

## Current Findings

So far, reinforcement learning policies trained with imitation pretraining and policy-gradient fine-tuning have performed competitively with, and in some scenarios better than, the hand-coded `avoid_wolves` heuristic.

In particular:
- RL policies can outperform `avoid_wolves` under default conditions
- the advantage appears to increase when grass becomes more scarce
- increasing network size did not meaningfully improve performance, suggesting that state representation and training setup matter more than pure model capacity

---

## Notes on Design Choices

### Discrete Actions
Although early ideas considered continuous headings, the current RL formulation uses discrete movement actions over the Moore neighborhood. This better matches the grid-based environment and simplifies policy-gradient training.

### Observation Space
The sheep’s RL observation currently includes:
- wolf presence in the visible neighborhood
- grass presence in the visible neighborhood
- the sheep’s current energy

### Reward Design
The most successful training so far has relied mostly on:
- reward for staying alive
- penalty for death

Additional shaping rewards were tested, but they had limited impact in the current setup.

---

## Future Directions

Possible extensions include:
- currently, sheep get the first-move advantage at each step; change this such that sheep and wolves effectively move simultaneously
- make wolves move faster than sheep
- negate the sheep ability to step through an oncoming wolf as an evasive maneuver (RL policy learned to exploit this trick)
- more advanced RL algorithms beyond vanilla policy gradient
- curriculum learning across map sizes and difficulty levels
- more formal statistical comparison of policies
- better visualization dashboards for policy evaluation

---

## Acknowledgments

This project is inspired by NetLogo’s **Wolf Sheep Predation** model by Uri Wilensky and extends that model into a Python-based experimental framework for studying learned and hand-coded survival strategies.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.