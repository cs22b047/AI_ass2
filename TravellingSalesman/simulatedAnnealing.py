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

def generate_neighbor(tour):
    a, b = random.sample(range(len(tour)), 2)
    new_tour = tour[:]
    new_tour[a], new_tour[b] = new_tour[b], new_tour[a]
    return new_tour


def simulated_annealing(cities, initial_temp=1000, cooling_rate=0.995, stop_temp=1e-3, max_iterations=50000):
    current_tour = list(range(len(cities)))
    random.shuffle(current_tour)
    current_cost = total_distance(current_tour, cities)
    best_tour = current_tour[:]
    best_cost = current_cost

    temp = initial_temp
    cost_progress = [current_cost]

    os.makedirs("frames_sa", exist_ok=True)

    def save_frame(tour, index):
        x = [cities[i][0] for i in tour + [tour[0]]]
        y = [cities[i][1] for i in tour + [tour[0]]]
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o-', color='purple', linewidth=2)
        for i, (xi, yi) in enumerate(cities):
            plt.text(xi + 0.5, yi + 0.5, str(i + 1), fontsize=8)
        plt.title(f"SA Iteration {index} | Cost: {round(total_distance(tour, cities), 2)}")
        plt.grid(True)
        plt.savefig(f"frames_sa/frame_{index:04d}.png")
        plt.close()

    save_frame(current_tour, 0)

    for iteration in range(1, max_iterations + 1):
        neighbor = generate_neighbor(current_tour)
        neighbor_cost = total_distance(neighbor, cities)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost
                save_frame(best_tour, iteration)

        cost_progress.append(best_cost)
        temp *= cooling_rate
        if temp < stop_temp:
            break

    return best_tour, best_cost, cost_progress


def plot_tour(tour, cities, filename):
    x = [cities[i][0] for i in tour + [tour[0]]]
    y = [cities[i][1] for i in tour + [tour[0]]]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', color='purple', linewidth=2)
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi + 0.5, yi + 0.5, str(i + 1), fontsize=8)
    plt.title("Final Simulated Annealing TSP Tour")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_cost(cost_progress, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(cost_progress, color='orange')
    plt.title("Best Cost Over Iterations (SA)")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def read_cities(filename):
    cities = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                _, x, y = parts
                cities.append((float(x), float(y)))
    return cities


def make_gif(output_file="tsp_sa.gif", frame_dir="frames_sa"):
    filenames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    images = [imageio.imread(os.path.join(frame_dir, f)) for f in filenames]
    imageio.mimsave(output_file, images, duration=0.4)


if __name__ == "__main__":
    random.seed(123)
    cities = read_cities("inp.txt")

    start = time.time()
    best_tour, best_cost, cost_progress = simulated_annealing(cities)
    end = time.time()
    duration = end - start

    print("Best Tour:")
    print([i + 1 for i in best_tour])
    print(f"Best Cost: {round(best_cost, 2)}")
    print(f"Running Time: {round(duration, 4)} seconds")

    plot_tour(best_tour, cities, "tsp_sa_tour.png")
    plot_cost(cost_progress, "sa_cost_progress.png")
    make_gif()

    print("Saved final tour as tsp_sa_tour.png")
    print("Saved cost graph as sa_cost_progress.png")
    print("GIF saved as tsp_sa.gif")
