from pathlib import Path
import pickle
import time

import matplotlib.pyplot as plt


class SolverBenchmarker:
    def __init__(self, parent_dir=None):
        self.metric_values = {}
        self.iteration_values = []
        self.wall_time_values = []
        self.start_time = None
        if parent_dir is not None:
            Path(parent_dir).mkdir(exist_ok=True, parents=True)
        self.parent_dir = Path(parent_dir) if parent_dir is not None else None

    def start_benchmark(self):
        self.start_time = time.time()

    def log_metric(self, metric_dict):
        iteration = len(self.iteration_values)
        wall_time = time.time() - self.start_time

        self.iteration_values.append(iteration)
        self.wall_time_values.append(wall_time)

        for metric_name, metric_value in metric_dict.items():
            if metric_name not in self.metric_values:
                self.metric_values[metric_name] = []

            self.metric_values[metric_name].append(metric_value)

            # print(f"Iteration {iteration}, {metric_name} {metric_value:.2f}, Wall Time {wall_time:.2f} s")

    def save_values(self):
        if self.parent_dir is None:
            return

        data_dict = {
            "metric_values": self.metric_values,
            "iteration_values": self.iteration_values,
            "wall_time_values": self.wall_time_values,
        }

        filename = self.parent_dir / "benchmark_state.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data_dict, f)

    def load_values(self):
        if self.parent_dir is None:
            return

        filename = self.parent_dir / "benchmark_state.pkl"
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} not found")

        with open(filename, "rb") as f:
            data_dict = pickle.load(f)

        self.metric_values = data_dict["metric_values"]
        self.iteration_values = data_dict["iteration_values"]
        self.wall_time_values = data_dict["wall_time_values"]

    def plot_metric_vs_iteration(self):
        if self.parent_dir is None:
            return
        else:
            filename = self.parent_dir / "iteration_plot.pdf"

        for metric_name, metric_values in self.metric_values.items():
            plt.plot(self.iteration_values, metric_values, label=metric_name)

        plt.xlabel("Iteration")
        plt.ylabel("Metric Value")
        plt.legend()
        if self.parent_dir is not None:
            plt.savefig(filename, format="pdf")

    def plot_metric_vs_wall_time(self):
        if self.parent_dir is None:
            return
        else:
            filename = self.parent_dir / "wall_time_plot.pdf"

        for metric_name, metric_values in self.metric_values.items():
            plt.plot(self.wall_time_values, metric_values, label=metric_name)

        plt.xlabel("Wall Time (seconds)")
        plt.ylabel("Metric Value")
        plt.legend()
        if self.parent_dir is not None:
            plt.savefig(filename, format="pdf")
