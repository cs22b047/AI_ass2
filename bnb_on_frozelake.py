import gymnasium as gym
import time
import heapq
import matplotlib.pyplot as plt

# Branch and Bound implementation
def branch_and_bound(env, max_time=600):
    start_time = time.time()
    state, _ = env.reset()
    frontier = [(0, [state])]
    visited = set()

    while frontier and (time.time() - start_time) < max_time:
        cost, path = heapq.heappop(frontier)
        current_state = path[-1]

        if current_state in visited:
            continue
        visited.add(current_state)

        env.unwrapped.s = current_state

        if current_state == env.observation_space.n - 1:
            return path, time.time() - start_time  # Reached goal

        for action in range(env.action_space.n):
            next_state, _, terminated, _, _ = env.step(action)

            if next_state not in path:
                heapq.heappush(frontier, (cost + 1, path + [next_state]))

    return None, time.time() - start_time

# Run multiple times
def test_runs(env, runs=5, max_time=600):
    times = []

    for i in range(runs):
        path, elapsed_time = branch_and_bound(env, max_time)
        if path is not None:
            print(f"Run {i+1}: Reached goal in {elapsed_time:.2f} seconds, Steps: {len(path)}")
        else:
            print(f"Run {i+1}: Goal not reached in {elapsed_time:.2f} seconds")
        times.append(elapsed_time)

    return times

# Main
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)  # Deterministic

    times = test_runs(env, runs=5, max_time=600)

    # Plotting
    plt.plot(range(1, len(times)+1), times, marker='o')
    plt.xlabel("Run")
    plt.ylabel("Time to reach goal (s)")
    plt.title("Branch and Bound on Frozen Lake")
    plt.grid(True)
    plt.show()

