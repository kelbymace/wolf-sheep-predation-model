import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=32, num_actions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state):
        return self.net(state)   # returns logits for 9 actions
        
def choose_action(policy_net, obs, greedy=False):

    state = torch.as_tensor(obs, dtype=torch.float32)
    logits = policy_net(state)
    dist = torch.distributions.Categorical(logits=logits)

    if greedy:
        action = torch.argmax(logits)
        log_prob = dist.log_prob(action)
    else:
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return int(action.item()), log_prob