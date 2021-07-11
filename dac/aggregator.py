from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer
from jsonrpclib import ServerProxy
import torch as T
import torch.nn.functional as F
import random
import numpy as np
from networks import CriticNetwork

handler = None
participants = []
conns = []
alpha_1 = 0.02
alpha_2 = 0.02
alpha_3 = 0.5
delta_1 = 50
delta_2 = 100
sigma_1 = 1.1
sigma_2 = 1.3


class Aggregator:
    def __init__(self, participants, input_size, action_size,beta=0.001):
        self.num_participants = len(participants)
        self.states = []
        self.input_size = input_size
        self.action_size = action_size
        self.critic = CriticNetwork(beta, input_size, n_actions=action_size)
        self.target_critic = CriticNetwork(beta, input_size, n_actions=action_size)
        self.gamma = 0.99
        self.record=True
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def update_network_parameters(self, tau=0.1):

        target_critic = self.target_critic.state_dict()
        value_params = self.critic.state_dict()

        target_value_state_dict = dict(target_critic)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_critic.load_state_dict(value_state_dict)

    def record_state_for_price(self, state):
        while not self.record:
            continue
        print(len(self.states))
        self.states.append(state)
        if len(self.states) == self.num_participants:
            self.record=False
            self.get_current_price()

    def get_current_price(self):
        L_t = 0
        for state in self.states:
            for i in range(1, 4):
                L_t += sum(state[i])
        lambda_t = alpha_1 * (L_t ** 2) + alpha_2 * L_t + alpha_3
        price = -1
        if lambda_t <= delta_1:
            price = lambda_t
        elif delta_1 < lambda_t <= delta_2:
            price = sigma_1 * lambda_t
        else:
            price = sigma_2 * lambda_t
        self.states = []
        # print(price)
        for conn in conns:
            conn.set_price(price)
        self.record=True
    def learn(self):
        conn = random.choice(conns)
        batch_idx = conn.get_batch_idx(500)
        action_data = []
        states_data = []
        next_states_data = []
        reward_data = []
        done_data = []
        actions = []
        next_actions=[]
        for conn in conns:
            state, action, reward, state_, done = conn.get_batch_data(batch_idx)
            curr_action = conn.get_action(state)
            next_action= conn.get_target_action(state_)
            if len(states_data) == 0:
                states_data = state
                action_data = action
                reward_data = reward
                next_states_data = state_
                done_data = done
                actions = curr_action
                next_actions = next_action
            else:
                states_data = np.hstack((states_data, state))
                next_states_data = np.hstack((next_states_data, state_))
                action_data = np.hstack((action_data, action))
                reward_data = [sum(i) for i in zip(reward_data, reward)]
                actions = np.hstack((actions, curr_action))
                next_actions = np.hstack((next_actions, next_action))
        print(np.array(states_data).shape)

        q = self.critic.forward(states_data, action_data).view(-1)
        print(f"q={q.shape}")
        q_next = self.target_critic.forward(next_states_data, next_actions).view(-1)
        print(f"q_n_1={q_next.shape}")
        q_next[done_data]=0
        print(f"q_n_2={q_next.shape}")
        self.critic.optimizer.zero_grad()
        q_hat = T.tensor(reward_data, dtype=T.float).to(self.device) + self.gamma * q_next
        print(f"r_shape={T.tensor(reward_data, dtype=T.float).shape}")
        print(f"q_hat={q_hat.shape}")
        critic_loss = 0.5 * F.mse_loss(q, q_hat)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.update_network_parameters()
        q = self.critic.forward(states_data, actions)
        critic_value = q.view(-1)

        return critic_value.tolist()


if __name__ == "__main__":
    with open('participants.txt', 'r') as participant:
        for line in participant:
            line = line.strip()
            if len(line) == 0:
                break
            ip, port,_ = line.split(" ")
            participants.append(f'http://{ip}:{port}')
    for p in participants:
        conns.append(ServerProxy(p))
    data_server = ServerProxy("http://127.0.0.1:8000")
    input_size, action_size = data_server.get_total_size()
    aggregator = Aggregator(participants,input_size, action_size)
    server = SimpleJSONRPCServer(('0.0.0.0', 8001))
    server.register_function(aggregator.get_current_price, 'get_price')
    server.register_function(aggregator.record_state_for_price, 'record_state')
    server.register_function(aggregator.learn, 'learn')
    server.serve_forever()
