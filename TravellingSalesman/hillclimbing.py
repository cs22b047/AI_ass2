import random
import math
import matplotlib.pyplot as plt
import time
import os
import imageio.v2 as imageio


def distance(city1, city2):
    return math.dist(city1, city2)

def total_distance(tour, cities):
    return sum(distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def generate_neighbor(tour):
    a, b = random.sample(range(len(tour)), 2)
    neighbor = tour[:]
    neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
    return neighbor

def hill_climbing_with_frames(cities, max_iterations=10000):
    current = random_tour(len(cities))
    current_cost = total_distance(current, cities)
    best = current
    best_cost = current_cost

    cost_progress = [current_cost]
    frames = []

    os.makedirs("frames", exist_ok=True)

    def save_frame(tour, index):
        x = [cities[i][0] for i in tour + [tour[0]]]
        y = [cities[i][1] for i in tour + [tour[0]]]
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o-', color='blue', linewidth=2)
        for i, (xi, yi) in enumerate(cities):
            plt.text(xi + 0.5, yi + 0.5, str(i + 1), fontsize=8)
        plt.title(f"Iteration {index} | Cost: {round(total_distance(tour, cities), 2)}")
        plt.grid(True)
        plt.savefig(f"frames/frame_{index:04d}.png")
        plt.close()

    save_frame(current, 0)

    for i in range(1, max_iterations + 1):
        neighbor = generate_neighbor(current)
        neighbor_cost = total_distance(neighbor, cities)
        if neighbor_cost < current_cost:
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current
                best_cost = current_cost
                save_frame(best, i)
        cost_progress.append(best_cost)

    return best, best_cost, cost_progress


def plot_final_tour(tour, cities, filename):
    x = [cities[i][0] for i in tour + [tour[0]]]
    y = [cities[i][1] for i in tour + [tour[0]]]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', color='blue', linewidth=2)
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi + 0.5, yi + 0.5, str(i + 1), fontsize=8)
    plt.title("Final Hill Climbing TSP Tour")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_cost_progress(cost_progress, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(cost_progress, color='green')
    plt.title("Best Cost Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def read_cities_from_file(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                _, x, y = parts
                cities.append((float(x), float(y)))
    return cities


def create_gif(output_file="tsp_hillclimb.gif", frames_dir="frames"):
    filenames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    images = [imageio.imread(os.path.join(frames_dir, f)) for f in filenames]
    imageio.mimsave(output_file, images, duration=0.4)

if __name__ == "__main__":
    random.seed(42)
    cities = read_cities_from_file("inp.txt")

    start_time = time.time()
    best_tour, best_cost, cost_progress = hill_climbing_with_frames(cities)
    end_time = time.time()
    runtime = end_time - start_time

    print("Best Tour: ")
    print([i + 1 for i in best_tour])
    print(f"Best Cost: {round(best_cost, 2)}")
    print(f"Running Time: {round(runtime, 4)} seconds")

    plot_final_tour(best_tour, cities, "tsp_tour.png")
    plot_cost_progress(cost_progress, "cost_progress.png")

    print("Saved tour plot as tsp_tour.png")
    print("Saved cost graph as cost_progress.png")

    create_gif()
    print("GIF saved as tsp_hillclimb.gif")
