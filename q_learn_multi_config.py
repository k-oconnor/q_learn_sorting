import numpy as np
import random
from collections import defaultdict
from itertools import product
from dataclasses import dataclass

@dataclass
class SorterConfig:
    """Configuration for the Q-learning sorter."""
    array_length: int = 10         # Length of arrays to sort
    num_episodes: int = 2000       # Number of training episodes
    gamma: float = 0.95           # Discount factor
    alpha: float = 0.1            # Learning rate
    epsilon: float = 0.1          # Exploration rate
    actions_per_step: int = 2     # Number of actions to take in each step
    max_steps_per_episode: int = 50  # Maximum steps per episode
    sorted_reward: float = 20.0    # Reward for achieving sorted state
    step_reward_multiplier: float = 2.0  # Multiplier for step-wise improvement

class QLearningSort:
    def __init__(self, config: SorterConfig):
        self.config = config
        self.Q = defaultdict(lambda: np.zeros(len(self.ACTION_COMBINATIONS)))
        self.initialize_actions()

    def initialize_actions(self):
        """Initialize basic actions and generate action combinations."""
        self.ACTIONS = [
            self.swap_adjacent,
            self.min_push,
            self.max_push,
            self.reverse_segment,
            self.cyclic_shift,
            self.swap_if_needed
        ]
        
        # Generate all possible action combinations
        self.ACTION_COMBINATIONS = list(product(
            range(len(self.ACTIONS)), 
            repeat=self.config.actions_per_step
        ))

    @staticmethod
    def swap_adjacent(arr):
        """Swap two adjacent elements."""
        idx = np.random.randint(len(arr) - 1)
        arr_copy = arr.copy()
        arr_copy[idx], arr_copy[idx + 1] = arr_copy[idx + 1], arr_copy[idx]
        return arr_copy
    
    @staticmethod
    def swap_if_needed(arr):
        """Swap two adjacent elements if they are out of order."""
        idx = np.random.randint(len(arr) - 1)
        if arr[idx] > arr[idx + 1]:  # Only swap if out of order
            arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
        return arr

    @staticmethod
    def min_push(arr):
        """Push minimum element to front."""
        arr_copy = arr.copy()
        min_idx = np.argmin(arr_copy)
        return np.insert(np.delete(arr_copy, min_idx), 0, arr_copy[min_idx])

    @staticmethod
    def max_push(arr):
        """Push maximum element to end."""
        arr_copy = arr.copy()
        max_idx = np.argmax(arr_copy)
        max_val = arr_copy[max_idx]
        return np.append(np.delete(arr_copy, max_idx), max_val)

    @staticmethod
    def reverse_segment(arr):
        """Reverse a random segment."""
        arr_copy = arr.copy()
        i, j = sorted(random.sample(range(len(arr_copy)), 2))
        arr_copy[i:j+1] = arr_copy[i:j+1][::-1]
        return arr_copy

    @staticmethod
    def cyclic_shift(arr):
        """Perform cyclic shift."""
        return np.roll(arr.copy(), np.random.choice([-1, 1]))

    @staticmethod
    def count_inversions(arr):
        """Count inversions in array."""
        return sum(1 for i in range(len(arr)) 
                  for j in range(i + 1, len(arr)) if arr[i] > arr[j])

    def choose_action_sequence(self, state):
        """Choose action sequence using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.config.epsilon:
            return random.randint(0, len(self.ACTION_COMBINATIONS) - 1)
        return np.argmax(self.Q[tuple(state)])

    def apply_action_sequence(self, state, action_sequence):
        """Apply sequence of actions to state."""
        current_state = np.array(state)
        states_visited = [tuple(current_state)]
        
        for action_idx in action_sequence:
            current_state = self.ACTIONS[action_idx](current_state)
            states_visited.append(tuple(current_state))
        
        return tuple(current_state), states_visited

    def calculate_reward(self, initial_state, final_state, intermediate_states):
        """Calculate reward for action sequence."""
        initial_inversions = self.count_inversions(initial_state)
        final_inversions = self.count_inversions(final_state)
        
        # Base reward based on improvement
        reward = self.config.step_reward_multiplier * (initial_inversions - final_inversions)
        
        # Penalty for visiting same state multiple times
        unique_states = len(set(intermediate_states))
        reward -= (len(intermediate_states) - unique_states)
        
        # Bonus for achieving sorted state
        if np.all(np.array(final_state)[:-1] <= np.array(final_state)[1:]):
            reward += self.config.sorted_reward
        
        return reward

    def train(self):
        """Train the Q-learning agent."""
        episode_rewards = []
        
        for episode in range(self.config.num_episodes):
            arr = np.random.permutation(self.config.array_length)
            state = tuple(arr)
            total_reward = 0
            
            for _ in range(self.config.max_steps_per_episode):
                action_seq_idx = self.choose_action_sequence(state)
                action_sequence = self.ACTION_COMBINATIONS[action_seq_idx]
                
                new_state, states_visited = self.apply_action_sequence(
                    state, action_sequence
                )
                
                reward = self.calculate_reward(state, new_state, states_visited)
                total_reward += reward
                
                # Q-learning update
                self.Q[state][action_seq_idx] += self.config.alpha * (
                    reward + 
                    self.config.gamma * np.max(self.Q[new_state]) - 
                    self.Q[state][action_seq_idx]
                )
                
                state = new_state
                if reward >= self.config.sorted_reward:
                    break
            
            episode_rewards.append(total_reward)
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    def test_policy(self, num_tests=5):
        """Test the learned policy."""
        for test in range(num_tests):
            arr = np.random.permutation(self.config.array_length)
            initial_state = tuple(arr)
            print(f"\nTest {test + 1}:")
            print(f"Initial state: {initial_state}")
            
            state = initial_state
            steps = 0
            
            while steps < self.config.max_steps_per_episode:
                action_seq_idx = np.argmax(self.Q[state])
                action_sequence = self.ACTION_COMBINATIONS[action_seq_idx]
                new_state, _ = self.apply_action_sequence(state, action_sequence)
                
                print(f"Step {steps + 1}:")
                print(f"Actions: {action_sequence}")
                print(f"New state: {new_state}")
                
                if np.all(np.array(new_state)[:-1] <= np.array(new_state)[1:]):
                    print("Sorted!")
                    break
                    
                state = new_state
                steps += 1


class SortingAnalyzer:
    """Analyzer for Q-learning sorting algorithm behavior."""
    
    def __init__(self, sorter: QLearningSort):
        self.sorter = sorter
        self.patterns = defaultdict(list)
        self.action_names = {
            0: "swap adjacent elements",
            1: "push minimum to front",
            2: "push maximum to end",
            3: "reverse segment",
            4: "cyclic shift",
            5: "swap if needed"
        }

    def action_to_string(self, action_idx):
        """Convert action index to descriptive string."""
        return self.action_names[action_idx]

    def analyze_basic_patterns(self, test_length=3):
        """Analyze basic patterns using arrays of specified length."""
        print(f"\nAnalyzing basic patterns with {test_length} elements:")
        print("-" * 40)
        
        # Generate all permutations of specified length
        test_arrays = list(product(range(test_length), repeat=test_length))
        
        for arr in test_arrays:
            if len(set(arr)) != test_length:  # Skip arrays with duplicates
                continue
                
            if np.all(np.array(arr)[:-1] <= np.array(arr)[1:]):  # Skip if already sorted
                continue
                
            state = tuple(arr)
            action_seq_idx = np.argmax(self.sorter.Q[state])
            actions = self.sorter.ACTION_COMBINATIONS[action_seq_idx]
            
            # Classify the pattern
            inversions = self.sorter.count_inversions(arr)
            pattern_key = f"{inversions}_inversions"
            
            if pattern_key not in self.patterns[len(arr)]:
                self.patterns[len(arr)].append(pattern_key)
                self._print_pattern_analysis(arr, actions, inversions)

    def analyze_action_frequencies(self):
        """Analyze frequency of action usage in learned policy."""
        action_frequencies = defaultdict(int)
        
        for state_values in self.sorter.Q.values():
            best_action_seq = self.sorter.ACTION_COMBINATIONS[np.argmax(state_values)]
            for action in best_action_seq:
                action_frequencies[action] += 1
        
        total_actions = sum(action_frequencies.values())
        if total_actions > 0:
            print("\nAction preferences:")
            for action, freq in sorted(
                action_frequencies.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                percentage = (freq / total_actions) * 100
                print(f"- {self.action_to_string(action)}: {percentage:.1f}%")

    def analyze_special_cases(self):
        """Analyze behavior on special case arrays."""
        print("\nSpecial cases analysis:")
        print("-" * 20)
        
        # Generate special cases based on array length
        special_cases = self._generate_special_cases()
        
        for case in special_cases:
            state = tuple(case)
            action_seq_idx = np.argmax(self.sorter.Q[state])
            actions = self.sorter.ACTION_COMBINATIONS[action_seq_idx]
            
            print(f"\nCase {case}:")
            self._print_action_sequence(actions)
            
            # Show result
            final_state, _ = self.sorter.apply_action_sequence(case, actions)
            print(f"Final state: {final_state}")

    def _generate_special_cases(self):
        """Generate special test cases based on array length."""
        n = self.sorter.config.array_length
        
        # Basic special cases
        cases = [
            tuple(range(n-1, -1, -1)),  # Completely reversed
            tuple([1, 0] + list(range(2, n))),  # Almost sorted with wrong prefix
        ]
        
        # Add pairs needing swaps
        if n >= 4:
            pairs_case = list(range(n))
            for i in range(0, n-1, 2):
                if i+1 < n:
                    pairs_case[i], pairs_case[i+1] = pairs_case[i+1], pairs_case[i]
            cases.append(tuple(pairs_case))
            
        return cases

    def _print_pattern_analysis(self, arr, actions, inversions):
        """Print analysis of a specific pattern."""
        print(f"\nPattern for {inversions} inversions:")
        print(f"Initial state: {arr}")
        self._print_action_sequence(actions)
        
        # Show result
        final_state, _ = self.sorter.apply_action_sequence(arr, actions)
        print(f"Final state: {final_state}")

    def _print_action_sequence(self, actions):
        """Print sequence of actions with descriptions."""
        print("Actions:")
        for idx, action in enumerate(actions, 1):
            print(f"  Step {idx}: {self.action_to_string(action)}")

    def run_full_analysis(self):
        """Run complete analysis of the sorting algorithm."""
        print("Analyzing learned sorting algorithm...\n")
        
        # Analyze basic patterns
        self.analyze_basic_patterns(min(3, self.sorter.config.array_length))
        
        print("\nFull algorithm analysis:")
        print("----------------------")
        
        # Analyze action frequencies
        self.analyze_action_frequencies()
        
        # Analyze special cases
        self.analyze_special_cases()


def main():
    # Create a sorter with specific configuration
    config = SorterConfig(
        array_length=5,
        num_episodes=1000,
        actions_per_step=2
    )
    
    # Initialize and train sorter
    sorter = QLearningSort(config)
    sorter.train()
    
    # Create analyzer and run analysis
    analyzer = SortingAnalyzer(sorter)
    analyzer.run_full_analysis()
    
    # Optional: Test the sorter after analysis
    print("\nTesting learned policy:")
    sorter.test_policy(num_tests=2)

if __name__ == "__main__":
    main()


def main():
    # Create config for different array lengths
    configs = [
        SorterConfig(array_length=3, num_episodes=500),
        SorterConfig(array_length=5, num_episodes=1000),
        SorterConfig(array_length=7, num_episodes=2000)
    ]
    
    for config in configs:
        print(f"\nTraining sorter for array length {config.array_length}")
        sorter = QLearningSort(config)
        sorter.train()
        print("\nTesting learned policy:")
        sorter.test_policy()

if __name__ == "__main__":
    main()