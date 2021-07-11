from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer
from jsonrpclib import ServerProxy
import argparse
import threading
from networks import ActorNetwork
from buffer import ReplayBuffer
import numpy as np
import torch as T

buffer = None
my_batch_idx = None
q_values = None
actor = None
log_probs = None
target_actor=None


def update_network_parameters(tau=0.1):
    target_actor_params = target_actor.state_dict()
    value_params = actor.state_dict()

    target_value_state_dict = dict(target_actor_params)
    value_state_dict = dict(value_params)

    for name in value_state_dict:
        value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                 (1 - tau) * target_value_state_dict[name].clone()

    target_actor.load_state_dict(value_state_dict)

def get_batch_idx(batch_size):
    max_mem = min(buffer.mem_cntr, buffer.mem_size)
    batch_idx = np.random.choice(max_mem, min(batch_size, max_mem))
    return batch_idx.tolist()


def get_batch_data(batch_idx):
    my_batch_idx = batch_idx
    states, actions, rewards, states_, dones = buffer.sample_buffer(batch_idx)
    return states.tolist(), actions.tolist(), rewards.tolist(), states_.tolist(), dones.tolist()


def get_action(obs):
    # print(len(obs))
    if len(obs) > 0:
        global log_probs
        actions, log_probs = actor.sample_normal(obs)
        # print(action.tolist())
        return actions.tolist()
    else:
        return []

def get_target_action(obs):
    # print(len(obs))
    if len(obs) > 0:
        actions, _= target_actor.sample_normal(obs)
        # print(action.tolist())
        return actions.tolist()
    else:
        return []


class ECC:
    def __init__(self, aggregator, data_server, buffer_server, ecc_id, max_episodes=1000):
        self.aggregator = aggregator
        self.data_server = data_server
        self.buffer_server = buffer_server
        self.max_episodes = max_episodes
        self.ecc_id = ecc_id
        self.price = -1
        self.beta = 720
        self.curr_eps_reward=0

    def set_price(self, price):
        self.price = price

    def run(self):
        with open(f'logs/reward_ecc_{self.ecc_id}.txt', 'w+') as r:
            r.write(f"ecc-{self.ecc_id}\n")
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        data = self.data_server.get_data(self.ecc_id)
        input_size = 1
        for app in data[0][1:]:
            for val in app:
                input_size += 2
        prev_left = [0 for i in range(int((input_size - 1) / 2))]
        input_size *= 2
        global buffer
        buffer = ReplayBuffer(1000000, [input_size], 1)
        prev_state = [0 for i in range(0, int(input_size / 2 - 1))]
        prev_price = 0
        prev_obs = []
        num_episodes = 0
        global actor
        actor = ActorNetwork(0.001, [input_size])
        global target_actor
        target_actor = ActorNetwork(0.001, [input_size])
        for counter, state in enumerate(data):
            self.price=-1
            print(f"Iteration-{counter}")
            self.aggregator.record_state(state)
            # print(state)
            # print(counter)
            while True:
                if self.price >= 0:
                    print(f'curr_price={self.price}')
                    break
            curr_state = []
            curr_time = state[0]
            z = 0
            for app in state[1:]:
                for val in app:
                    curr_state.append(curr_time)
                    curr_state.append(val + prev_left[z])
                    z += 1
            obs = [prev_price, self.price]
            obs = obs + prev_state
            obs = obs + curr_state
            if len(prev_obs) > 0:
                buffer.store_transition(prev_obs, action, reward, obs, False)
                self.curr_eps_reward+=reward
            # print(obs)
            actor.eval()
            action = actor.sample_normal([obs])[0].item()
            actor.train()
            print(f"action={action}")
            action_left = action
            z = 0
            for app in state[1:]:
                for val in app:
                    if action_left > 0:
                        prev_left[z] = 0 if action_left > val else val - action_left
                    else:
                        prev_left[z] = val
                    action_left = action - val if action_left > val else 0
                    z += 1
            reward = -action * self.price
            # This if condition is own addition,don't search in paper
            # if action_left > 0:
            #     reward *= 2
            prev_price = self.price
            prev_state = curr_state
            prev_obs = obs

            if curr_time == 144:
                num_episodes += 1
                print(f"num_episodes={num_episodes}")
                left_out = sum(prev_left)
                if left_out == 0:
                    reward = reward + 50
                else:
                    reward = reward - 60 * left_out
                buffer.store_transition(obs, action, reward, [0 for item in obs], True)
                self.curr_eps_reward+=reward
                with open(f'logs/reward_ecc_{self.ecc_id}.txt', 'a') as r:
                    print(f"reward={self.curr_eps_reward}")
                    r.write(f'{self.curr_eps_reward}\n')
                self.curr_eps_reward=0
                prev_state = [0 for i in range(0, int(input_size / 2 - 1))]
                prev_obs = []
                prev_price = 0
                prev_left = [0 for i in range(int((input_size - 1) / 2))]

            if (counter + 1) % self.beta == 0 or num_episodes == self.max_episodes:
                print("Learning")
                global log_probs
                q_value = self.aggregator.learn()
                actor_loss = -log_probs.view(-1) *T.tensor(q_value).to(device)
                actor_loss = T.mean(actor_loss)
                actor.optimizer.zero_grad()
                actor_loss.backward()
                actor.optimizer.step()
                update_network_parameters()
                if num_episodes == self.max_episodes:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", help="Enter Port Number")
    parser.add_argument("-I", "--id")
    args = parser.parse_args()
    port = int(args.port)
    id = int(args.id)
    print(port)
    aggregator_ip = "http://127.0.0.1:8001"
    aggregator = ServerProxy(aggregator_ip)
    data_ip = "http://127.0.0.1:8000"
    data_server = ServerProxy(data_ip)
    buffer_ip = "http://127.0.0.1:8002"
    buffer_server = ServerProxy(buffer_ip)
    ecc = ECC(aggregator, data_server, buffer_server, id)
    t = threading.Thread(target=ecc.run, args=())
    t.start()
    server = SimpleJSONRPCServer(('0.0.0.0', port))
    server.register_function(ecc.set_price)
    server.register_function(get_batch_idx)
    server.register_function(get_batch_data)
    server.register_function(get_action)
    server.register_function(get_target_action)
    server.serve_forever()
