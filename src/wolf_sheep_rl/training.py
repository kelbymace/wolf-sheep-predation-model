import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .policy import PolicyNetwork
from .model import WolfSheepModel

def generate_expert_data(num_samples=5000, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}

    states = []
    actions = []

    expert_kwargs = dict(model_kwargs)
    expert_kwargs["sheep_strategy"] = "avoid_wolves"
    expert_kwargs["wolf_strategy"] = "seek_sheep"

    for _ in range(num_samples):
        model = WolfSheepModel(**expert_kwargs)
        model.setup()

        if len(model.sheep) != 1:
            raise ValueError("Behavior cloning expects exactly one sheep.")

        sheep = model.sheep[0]
        obs = model.get_sheep_observation(sheep)
        action = sheep.get_avoid_wolves_action()

        states.append(obs)
        actions.append(action)

    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long)
    )

def pretrain_policy_with_expert(policy_net, num_samples=5000, batch_size=64, num_epochs=10, lr=1e-3, model_kwargs=None):
    states, actions = generate_expert_data(num_samples=num_samples, model_kwargs=model_kwargs)

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []

    policy_net.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_states, batch_actions in loader:
            logits = policy_net(batch_states)
            loss = loss_fn(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Pretrain epoch {epoch + 1}/{num_epochs}, loss = {avg_loss:.4f}")

    return policy_net, losses

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train_policy_gradient(num_episodes=1000, gamma=0.99, learning_rate=1e-3, max_steps=200, model_kwargs=None, pretrain_with_expert=False, pretrain_samples=5000, pretrain_epochs=10, pretrain_lr=1e-3):

    model_kwargs = dict(model_kwargs or {})
    sight_radius = model_kwargs.get("sheep_sight_radius", 2)
    cells = (2 * sight_radius + 1) ** 2
    input_dim = cells * 2 + 1 # 2x for wolf and grass preesence, +1 for the current energy level

    policy_net = PolicyNetwork(input_dim=input_dim)

    if pretrain_with_expert:
        policy_net, pretrain_losses = pretrain_policy_with_expert(
            policy_net,
            num_samples=pretrain_samples,
            num_epochs=pretrain_epochs,
            lr=pretrain_lr,
            model_kwargs=model_kwargs,
        )

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    default_kwargs = {
        "width": 25,
        "height": 25,
        "initial_number_sheep": 1,
        "initial_number_wolves": 5,
        "model_version": "rl-training",
        "sheep_strategy": "rl",
        "wolf_strategy": "seek_sheep",
        "enable_grass": True,
        "grass_regrowth_time": 30,
        "policy_net": policy_net,
        "sheep_sight_radius": sight_radius,
        "wolf_sight_radius": model_kwargs.get("wolf_sight_radius", 2),
    }
    default_kwargs.update(model_kwargs)

    episode_lengths = []
    best_avg_len = 0 # Keep track of the best policy during training and save that one

    for episode in range(num_episodes):
        model = WolfSheepModel(**default_kwargs)
        model.collect_log_probs = True
        model.setup()

        rewards = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            reward, done = model.go()
            rewards.append(reward)
            steps += 1

        returns = compute_returns(rewards, gamma=gamma)
        loss = 0
        for log_prob, G in zip(model.current_episode_log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_lengths.append(steps)

        if (episode + 1) % 50 == 0:
            avg_len = sum(episode_lengths[-50:]) / min(50, len(episode_lengths))
            print(f"Episode {episode + 1}: avg length over last 50 = {avg_len:.2f}")

            if avg_len > best_avg_len:
                best_avg_len = avg_len
                torch.save(policy_net.state_dict(), "best_sheep_policy_2.0.pt")

    return policy_net, episode_lengths


def load_policy(path, input_dim, hidden_dim=32, num_actions=8):
    policy_net = PolicyNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)
    policy_net.load_state_dict(torch.load(path, map_location="cpu"))
    policy_net.eval()
    return policy_net