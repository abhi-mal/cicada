import h5py
import numpy as np
import numpy.typing as npt

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import data
from typing import List, Tuple
import matplotlib.pyplot as plt

class RegionETGenerator:
    def __init__(
        self, train_size: float = 0.5, val_size: float = 0.1, test_size: float = 0.4
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = 42

    def get_generator(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        batch_size: int,
        drop_remainder: bool = False,
    ) -> data.Dataset:
        dataset = data.Dataset.from_tensor_slices((X, y))
        return (
            dataset.shuffle(210 * batch_size)
            .batch(batch_size, drop_remainder=drop_remainder)
            .prefetch(data.AUTOTUNE)
        )

    def get_data(self, datasets_paths: List[Path]) -> npt.NDArray:
        inputs = []
        for dataset_path in datasets_paths:
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions"][:].astype("float32")
            )
        X = np.concatenate(inputs)
        X = np.reshape(X, (-1, 18, 14, 1))
        return X

    def get_data_split(
        self, datasets_paths: List[Path]
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        X = self.get_data(datasets_paths)
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        X_train, X_val = train_test_split(
            X_train,
            test_size=self.val_size / (self.val_size + self.train_size),
            random_state=self.random_state,
        )
        return (X_train, X_val, X_test)

    def get_benchmark(
        self, datasets: dict, filter_acceptance=True
    ) -> Tuple[dict, list]:
        signals = {}
        acceptance = []
        for dataset in datasets:
            if not dataset["use"]:
                continue
            signal_name = dataset["name"]
            for dataset_path in dataset["path"]:
                X = h5py.File(dataset_path, "r")["CaloRegions"][:].astype("float32")
                X = np.reshape(X, (-1, 18, 14, 1))
                try:
                    flags = h5py.File(dataset_path, "r")["AcceptanceFlag"][:].astype(
                        "bool"
                    )
                    fraction = 100.0#np.round(100 * sum(flags) / len(flags), 2)
                except KeyError:
                    fraction = 100.0
                if filter_acceptance:
                    X = X[flags]
                signals[signal_name] = X
                acceptance.append({"signal": signal_name, "acceptance": fraction})
        return signals, acceptance

    def generate_random_exposure_data(
        self,
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        num_samples_train: int = 100000,
        num_samples_val: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates two random numpy arrays with values in the range of the min and max
        values of the provided input arrays X_train and X_val.

        Args:
             X_train (np.ndarray): First input array.
             X_val (np.ndarray): Second input array.
             num_samples (int): Number of random samples to generate for each array.

        Returns:
             tuple:  Two numpy arrays of shape (num_samples, 18, 14, 1) filled with random values
                     in the range of the min and max of X_train and X_val.
        """
        # Find the combined min and max values from X_train and X_val
        global_min = min(np.min(X_train), np.min(X_val))
        global_max = max(np.max(X_train), np.max(X_val))
        # Generate two random arrays with values in the global range
        rand_train = np.random.uniform(global_min, global_max, size=(num_samples_train, 18, 14, 1)).astype("float32")
        rand_val = np.random.uniform(global_min, global_max, size=(num_samples_val, 18, 14, 1)).astype("float32")
        return rand_train, rand_val

    def generate_random_exposure_data_from_hist(
        self,
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        num_samples_train: int = 100000,
        num_samples_val: int = 100000,
        plot_hist: bool = True,
        hist_path: str = 'exposure_histogram.png'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates random numpy arrays by sampling from the 1d distribution of the
        cell values of the provided input arrays X_train and X_val.

        It creates a 1D histogram from the combined cell values of X_train and X_val. 
        It then samples from this distribution to generate new data, ensuring the 
        new data statistically resembles the original.

        Args:
                X_train (np.ndarray): First input array.
                X_val (np.ndarray): Second input array.
                num_samples_train (int): Number of random samples to generate for the training set.
                num_samples_val (int): Number of random samples to generate for the validation set.
                plot_hist (bool): If True, a plot of the histogram will be generated and saved.
                hist_path (str): The file path where the histogram plot will be saved.

        Returns:
                tuple: Two numpy arrays, rand_train and rand_val, of shapes 
                        (num_samples_train, 18, 14, 1) and (num_samples_val, 18, 14, 1) 
                        respectively, filled with values sampled from the input data's distribution.
        """
        # Combine and flatten the input data to get a 1D array of all cell values
        combined_data = np.concatenate((X_train.flatten(), X_val.flatten()))

        # Create histogram to sample from
        # np.histogram returns the frequency counts and the bin edges.
        counts, bin_edges = np.histogram(combined_data, bins=256, density=False)

        # The representative value is the center of the bin.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Convert the frequency counts into a probability distribution.
        probabilities = counts / counts.sum()
        print(bin_centers)
        print(probabilities)

        # If requested, plot the histogram and save it
        if plot_hist:
                print(f"Plotting histogram and saving to {hist_path}...")
                plt.figure(figsize=(12, 7))
                # We use a bar plot to visualize the calculated histogram.
                # The width of the bars is set to the width of a single bin.
                bar_width = bin_edges[1] - bin_edges[0]
                plt.bar(bin_centers, counts, width=bar_width, align='center', edgecolor='black', alpha=0.8)
                
                plt.title("Histogram of Cell Values in Combined Trainign and Validation Data", fontsize=16)
                plt.xlabel("Cell Value", fontsize=12)
                plt.ylabel("Frequency (Number of Cells)", fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()
                print("Histogram saved.")

        # Generate random samples using the calculated distribution.
        print("Generating random samples for training set...")
        total_train_elements = num_samples_train * X_train.shape[1] * X_train.shape[2]
        random_samples_train_flat = np.random.choice(
                bin_centers,
                size=total_train_elements,
                p=probabilities
        )
        # Reshape the flat array to match X_train
        rand_train = random_samples_train_flat.reshape(
                (num_samples_train, X_train.shape[1], X_train.shape[2], 1)
        ).astype("float32")

        print("Generating random samples for validation set...")
        total_val_elements = num_samples_val * X_val.shape[1] * X_val.shape[2]
        random_samples_val_flat = np.random.choice(
                bin_centers,
                size=total_val_elements,
                p=probabilities
        )
        # Reshape the flat array to match X_val
        rand_val = random_samples_val_flat.reshape(
                (num_samples_val, X_val.shape[1], X_val.shape[2], 1)
        ).astype("float32")
        
        print("Data generation complete.")
        return rand_train, rand_val