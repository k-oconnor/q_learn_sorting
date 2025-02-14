import numpy as np
import random
from collections import defaultdict

# Parameters
num_episodes = 1000  # Number of training episodes
gamma = 0.95  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate

def count_inversions(arr):
    """Count number of inversions in an array (how far it is from sorted)."""
    return sum(1 for i in range(len(arr)) for j in range(i + 1, len(arr)) if arr[i] > arr[j])

# Define available actions
def swap_adjacent(arr):
    idx = np.random.randint(len(arr) - 1)
    arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
    return arr

def min_push(arr):
    min_idx = np.argmin(arr)
    arr = np.insert(np.delete(arr, min_idx), 0, arr[min_idx])
    return arr

def max_push(arr):
    max_idx = np.argmax(arr)
    max_val = arr[max_idx]
    arr = np.delete(arr, max_idx)  # Remove the max element
    arr = np.append(arr, max_val)  # Append it to the end
    return arr

def reverse_segment(arr):
    i, j = sorted(random.sample(range(len(arr)), 2))
    arr[i:j+1] = arr[i:j+1][::-1]
    return arr

def cyclic_shift(arr):
    return np.roll(arr, np.random.choice([-1, 1]))

def get_rand_idx(arr):
    return np.random.randint(0, len(arr)-1)

ACTIONS = [swap_adjacent, min_push, max_push, reverse_segment, cyclic_shift]

# Q-table (State-action value store)
Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

def choose_action(state):
    """Epsilon-greedy policy for action selection."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0,4)  # Explore
    return np.argmax(Q[tuple(state)])  # Exploit

def train_q_learning():
    for episode in range(num_episodes):
        arr = np.random.permutation(5)  # Random unsorted array
        state = tuple(arr)
        print(state)
        
        for _ in range(100):  # Limit steps per episode
            action_idx = choose_action(state)
            new_arr = ACTIONS[action_idx](np.array(state))
            new_state = tuple(new_arr)
            
            # Compute reward
            reward = 1 if count_inversions(new_arr) < count_inversions(arr) else -1

            if np.all(new_arr[:-1] <= new_arr[1:]):
                reward += 10  # Fully sorted bonus
            
            # Q-learning update rule
            Q[state][action_idx] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action_idx])
            
            # Transition
            state = new_state
            if reward >= 10:  # If sorted, stop early
                break
        
        if episode % 1000 == 0:
            print(f"Episode {episode} complete.")

# Train agent
train_q_learning()


# Print learned policy
def print_policy():
    print("Learned Q-values for sorting:")
    for state, actions in Q.items():
        print(f"State {state}: Best Action {np.argmax(actions)}")

print_policy()


# Test trained policy
def test_q_learning():
    arr = np.random.permutation(5)  # Random unsorted array
    print("Initial state:", arr)
    
    for _ in range(100):
        action_idx = np.argmax(Q[tuple(arr)])
        arr = ACTIONS[action_idx](np.array(arr))
        print("New state:", arr)
        if np.all(arr[:-1] <= arr[1:]):
            print("Sorted!")
            break

test_q_learning()
