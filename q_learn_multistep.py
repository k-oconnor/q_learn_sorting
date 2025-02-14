import numpy as np
import random
from collections import defaultdict
from itertools import product

# Parameters
num_episodes = 1000  # Number of training episodes
gamma = 0.95  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate
actions_per_step = 2  # Number of actions to take in each step

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
    arr = np.delete(arr, max_idx)
    arr = np.append(arr, max_val)
    return arr

def reverse_segment(arr):
    i, j = sorted(random.sample(range(len(arr)), 2))
    arr[i:j+1] = arr[i:j+1][::-1]
    return arr

def cyclic_shift(arr):
    return np.roll(arr, np.random.choice([-1, 1]))

ACTIONS = [swap_adjacent, min_push, max_push, reverse_segment, cyclic_shift]

# Generate all possible action combinations
ACTION_COMBINATIONS = list(product(range(len(ACTIONS)), repeat=actions_per_step))

# Q-table (State-action value store)
Q = defaultdict(lambda: np.zeros(len(ACTION_COMBINATIONS)))

def apply_action_sequence(state, action_sequence):
    """Apply a sequence of actions to a state."""
    current_state = np.array(state)
    states_visited = [tuple(current_state)]
    
    for action_idx in action_sequence:
        current_state = ACTIONS[action_idx](current_state.copy())
        states_visited.append(tuple(current_state))
    
    return tuple(current_state), states_visited

def choose_action_sequence(state):
    """Epsilon-greedy policy for action sequence selection."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(ACTION_COMBINATIONS) - 1)
    return np.argmax(Q[tuple(state)])

def calculate_reward(initial_state, final_state, intermediate_states):
    """Calculate reward for a sequence of actions."""
    initial_inversions = count_inversions(initial_state)
    final_inversions = count_inversions(final_state)
    
    # Base reward based on improvement
    reward = 2 * (initial_inversions - final_inversions)
    
    # Penalty for visiting the same state multiple times
    unique_states = len(set(intermediate_states))
    reward -= (len(intermediate_states) - unique_states)
    
    # Bonus for achieving sorted state
    if np.all(np.array(final_state)[:-1] <= np.array(final_state)[1:]):
        reward += 20
    
    return reward

def train_q_learning():
    episode_rewards = []
    
    for episode in range(num_episodes):
        arr = np.random.permutation(5)  # Random unsorted array
        state = tuple(arr)
        total_reward = 0
        
        for _ in range(50):  # Limit steps per episode
            action_seq_idx = choose_action_sequence(state)
            action_sequence = ACTION_COMBINATIONS[action_seq_idx]
            
            # Apply the sequence of actions
            new_state, states_visited = apply_action_sequence(state, action_sequence)
            
            # Calculate reward
            reward = calculate_reward(state, new_state, states_visited)
            total_reward += reward
            
            # Q-learning update rule
            Q[state][action_seq_idx] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state][action_seq_idx]
            )
            
            # Transition
            state = new_state
            if reward >= 20:  # If sorted, stop early
                break
        
        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

def print_policy():
    """Print the learned policy for some example states."""
    print("\nLearned Policy Examples:")
    test_states = [
        tuple(np.random.permutation(5)) for _ in range(5)
    ]
    
    for state in test_states:
        best_action_seq_idx = np.argmax(Q[state])
        best_actions = ACTION_COMBINATIONS[best_action_seq_idx]
        print(f"State {state}: Best Action Sequence {best_actions}")

def test_policy():
    """Test the learned policy on a random initial state."""
    arr = np.random.permutation(5)
    initial_state = tuple(arr)
    print(f"\nTesting Policy:")
    print(f"Initial state: {initial_state}")
    
    state = initial_state
    steps = 0
    
    while steps < 20:  # Limit maximum steps
        action_seq_idx = np.argmax(Q[state])
        action_sequence = ACTION_COMBINATIONS[action_seq_idx]
        new_state, _ = apply_action_sequence(state, action_sequence)
        
        print(f"Step {steps + 1}:")
        print(f"Actions: {action_sequence}")
        print(f"New state: {new_state}")
        
        if np.all(np.array(new_state)[:-1] <= np.array(new_state)[1:]):
            print("Sorted!")
            break
            
        state = new_state
        steps += 1

# Train and test
train_q_learning()
print_policy()
test_policy()

def action_to_string(action_idx):
    """Convert action index to descriptive string."""
    action_names = {
        0: "swap adjacent elements",
        1: "push minimum to front",
        2: "push maximum to end",
        3: "reverse segment",
        4: "cyclic shift"
    }
    return action_names[action_idx]

def analyze_sorting_algorithm():
    """Analyze and print the learned sorting algorithm."""
    print("Analyzing learned sorting algorithm...\n")
    
    # Test on all permutations of length 3 first to understand basic patterns
    test_arrays = list(product(range(3), repeat=3))
    patterns = defaultdict(list)
    
    print("Basic patterns with 3 elements:")
    print("------------------------------")
    
    for arr in test_arrays:
        if len(set(arr)) != 3:  # Skip arrays with duplicates
            continue
            
        if np.all(np.array(arr)[:-1] <= np.array(arr)[1:]):  # Skip if already sorted
            continue
            
        state = tuple(arr)
        action_seq_idx = np.argmax(Q[state])
        actions = ACTION_COMBINATIONS[action_seq_idx]
        
        # Classify the pattern
        inversions = count_inversions(arr)
        pattern_key = f"{inversions}_inversions"
        
        if pattern_key not in patterns[len(arr)]:
            patterns[len(arr)].append(pattern_key)
            print(f"\nPattern for {inversions} inversions:")
            print(f"Initial state: {arr}")
            print("Actions:")
            for idx, action in enumerate(actions, 1):
                print(f"  Step {idx}: {action_to_string(action)}")
            
            # Show result
            final_state, _ = apply_action_sequence(arr, actions)
            print(f"Final state: {final_state}")
    
    print("\nFull algorithm analysis:")
    print("----------------------")
    print("General strategy:")
    
    # Analyze Q-values to find most commonly used actions
    action_frequencies = defaultdict(int)
    for state_values in Q.values():
        best_action_seq = ACTION_COMBINATIONS[np.argmax(state_values)]
        for action in best_action_seq:
            action_frequencies[action] += 1
    
    total_actions = sum(action_frequencies.values())
    if total_actions > 0:  # Avoid division by zero
        print("\nAction preferences:")
        for action, freq in sorted(action_frequencies.items(), key=lambda x: x[1], reverse=True):
            percentage = (freq / total_actions) * 100
            print(f"- {action_to_string(action)}: {percentage:.1f}%")
    
    # Analyze special cases
    print("\nSpecial cases:")
    special_cases = [
        (4, 3, 2, 1, 0),  # Completely reversed
        (1, 0, 2, 3, 4),  # Almost sorted with wrong prefix
        (0, 2, 1, 4, 3),  # Pairs need swapping
    ]
    
    for case in special_cases:
        state = tuple(case)
        action_seq_idx = np.argmax(Q[state])
        actions = ACTION_COMBINATIONS[action_seq_idx]
        
        print(f"\nCase {case}:")
        print("Actions:")
        for idx, action in enumerate(actions, 1):
            print(f"  Step {idx}: {action_to_string(action)}")
        
        # Show result
        final_state, _ = apply_action_sequence(case, actions)
        print(f"Final state: {final_state}")

# Run the analysis
analyze_sorting_algorithm()