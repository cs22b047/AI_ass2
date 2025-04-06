import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

# Heuristic: Manhattan distance on FrozenLake grid
def manhattan_distance(state, goal, grid_size):
    x1, y1 = divmod(state, grid_size)
    x2, y2 = divmod(goal, grid_size)
    return abs(x1 - x2) + abs(y1 - y2)

def ida_star(env, max_time=600):
    start_time = time.time()
    start_state, _ = env.reset()
    goal_state = env.observation_space.n - 1
    grid_size = int(np.sqrt(env.observation_space.n))

    def search(path, g, bound):
        current_state = path[-1]
        f = g + manhattan_distance(current_state, goal_state, grid_size)

        if f > bound:
            return f, None
        if current_state == goal_state:
            return True, path

        min_threshold = float('inf')

        for action in range(env.action_space.n):
            env.unwrapped.s = current_state
            next_state, _, terminated, _, _ = env.step(action)

            if next_state in path:
                continue

            path.append(next_state)
            result, found_path = search(path, g + 1, bound)

            if result is True:
                return True, found_path
            if result < min_threshold:
                min_threshold = result
            path.pop()

        return min_threshold, None

    bound = manhattan_distance(start_state, goal_state, grid_size)
    path = [start_state]

    while (time.time() - start_time) < max_time:
        result, found_path = search(path, 0, bound)
        if result is True:
            return found_path, time.time() - start_time
        if result == float('inf'):
            break
        bound = result

    return None, time.time() - start_time

def test_ida_runs(env, runs=5, max_time=600):
    times = []

    for i in range(runs):
        path, elapsed_time = ida_star(env, max_time)
        if path:
            print(f"Run {i+1}: Reached goal in {elapsed_time:.2f}s | Steps: {len(path)}")
        else:
            print(f"Run {i+1}: Goal not reached in {elapsed_time:.2f}s")
        times.append(elapsed_time)

    return times

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    times = test_ida_runs(env)

    plt.plot(range(1, len(times)+1), times, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Time to reach goal (s)")
    plt.title("IDA* on Frozen Lake")
    plt.grid(True)
    plt.show()
