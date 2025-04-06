import gymnasium as gym
import time
import heapq
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio


def save_gif_from_path(env_id, path, gif_name, is_slippery=False):
    env = gym.make(env_id, is_slippery=is_slippery, render_mode="rgb_array")
    frames = []

    obs, _ = env.reset()
    frames.append(env.render())

    for i in range(1, len(path)):
        curr_state = path[i - 1]
        next_state = path[i]

        # Find action that gets from curr_state to next_state
        for action in range(env.action_space.n):
            for prob, s, _, _ in env.unwrapped.P[curr_state][action]:
                if s == next_state and prob > 0:
                    obs, _, terminated, truncated, _ = env.step(action)
                    frames.append(env.render())
                    break
            else:
                continue
            break

    env.close()

    imageio.mimsave(gif_name, frames, duration=0.5)
    print(f"Saved GIF: {gif_name}")
# Branch and Bound implementation
def branch_and_bound(env, max_time=600):
    import heapq, time
    start_time = time.time()
    start_state, _ = env.reset()
    goal_state = env.observation_space.n - 1
    frontier = [(0, [start_state])]
    visited = set()

    while frontier and (time.time() - start_time) < max_time:
        cost, path = heapq.heappop(frontier)
        current_state = path[-1]

        if current_state == goal_state:
            return path, time.time() - start_time

        if current_state in visited:
            continue
        visited.add(current_state)

        for action in range(env.action_space.n):
            for prob, next_state, reward, done in env.unwrapped.P[current_state][action]:
                if prob > 0 and next_state not in path:
                    heapq.heappush(frontier, (cost + 1, path + [next_state]))

    return None, time.time() - start_time
# Run multiple times
def test_runs(env, runs=5, max_time=600):
    times = []
    grid_size = int(np.sqrt(env.observation_space.n))

    for i in range(runs):
        path, elapsed_time = branch_and_bound(env, max_time)
        if path:
            print(f"[BnB] Run {i+1}: Reached goal in {elapsed_time:.2f}s | Steps: {len(path)}")
            save_gif_from_path("FrozenLake-v1", path, f"bnb_run{i+1}.gif", is_slippery=False)
        else:
            print(f"[BnB] Run {i+1}: Goal not reached in {elapsed_time:.2f}s")
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

