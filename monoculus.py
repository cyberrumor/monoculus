#!/usr/bin/env python3
import argparse
import time

import curses
import numpy as np
import torch

class SOM2D:
    def __init__(self, width, height, lr=0.1, sigma=3.0):
        self.width = width
        self.height = height
        self.lr = lr
        self.sigma = sigma
        self.weights = None

    def _neighborhood(self, winner_x, winner_y):
        """Return 2D gaussian centered at winner location"""
        y = torch.arange(0, self.height).view(-1, 1)
        x = torch.arange(0, self.width).view(1, -1)
        dist_sq = (x - winner_x)**2 + (y - winner_y)**2
        return torch.exp(-dist_sq / (2 * self.sigma**2))  # shape [H, W]

    def train_step(self, input_vec):
        input_vec = input_vec.view(-1)  # Ensure shape [D]

        # Lazy initialization of weights
        if self.weights is None:
            input_dim = input_vec.shape[0]
            self.weights = torch.randn(self.height, self.width, input_dim)

        # Check for input/weight shape mismatch
        assert input_vec.shape[0] == self.weights.shape[2], (
            f"Input dimension mismatch: got {input_vec.shape[0]}, expected {self.weights.shape[2]}"
        )

        # Compute distances to input vector
        dists = torch.norm(self.weights - input_vec.view(1, 1, -1), dim=2)
        flat_idx = torch.argmin(dists).item()
        winner_y, winner_x = divmod(flat_idx, self.width)

        # Neighborhood influence
        influence = self._neighborhood(winner_x, winner_y).unsqueeze(2)  # [H, W, 1]
        delta = self.lr * influence * (input_vec.view(1, 1, -1) - self.weights)
        self.weights += delta

        return winner_x, winner_y


    def get_activation_map(self, input_vec):
        dists = torch.norm(self.weights - input_vec.view(1, 1, -1), dim=2)
        normed = 1 - dists / dists.max()  # invert and normalize
        return normed


def tokenize_text(text: str) -> (list[torch.Tensor], set[str]):
    """
    Character-level embedding, simple and universal.
    """
    tokens = list(text)
    vocab = sorted(set(tokens))
    stoi = {c: i for i, c in enumerate(vocab)}
    identity_matrix = torch.eye(len(vocab))
    encoded = [identity_matrix[stoi[c]] for c in tokens]
    return encoded, vocab


def draw_grid(stdscr, act_map, winner=None):
    stdscr.clear()
    for y in range(act_map.shape[0]):
        row = ""
        for x in range(act_map.shape[1]):
            val = act_map[y, x].item()

            # Represent the floats in the tensors as
            # characters so the activity is easier to visualize.

            # straight lines only
            # chars = ["#", "H", "=", "-", " "]

            # digital rain
            # chars = ["|", "!", ";", ":", "'", ",", ".", " "]

            # density
            chars = ["@", "0", "G", "Q", "O", "C", "o", "c", ";", ":", ",", ".", " "]

            # Generate thresholds exponentially spaced between 0 and 1 (excluding 1)
            # Small values spaced out, large values bunched near 1
            num_levels = len(chars)
            thresholds = np.linspace(1.0 / num_levels, 1.0 - (1.0 / num_levels), num=num_levels)[::-1]
            char_thresholds = dict(zip(chars, thresholds))

            char = " "
            for k, v in char_thresholds.items():
                if val > v:
                    char = k
                    break

            if winner and (x, y) == winner:
                char = "*"
            row += f"{char:<2}"
        stdscr.addstr(y, 0, row)
    stdscr.refresh()


def train_from_file(filepath, som, pause=True):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    data, vocab = tokenize_text(text)

    def curses_main(stdscr):
        # Draw the plain network before any input.
        # draw_grid(stdscr, act_map, winner=None)

        for i, vec in enumerate(data):
            winner_x, winner_y = som.train_step(vec)
            act_map = som.get_activation_map(vec)
            draw_grid(stdscr, act_map, winner=(winner_x, winner_y))

            stdscr.addstr(som.height + 1, 0, f"Step {i + 1}/{len(data)} — Token: '{vocab[torch.argmax(vec).item()]}'")
            stdscr.refresh()

            if pause:
                key = stdscr.getkey()
                if key == "q":
                    break
            else:
                time.sleep(0.1)

    curses.wrapper(curses_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Plain text file to train on")
    parser.add_argument("--nopause", action="store_true", help="Run automatically without pausing")
    args = parser.parse_args()

    som = SOM2D(width=40, height=40)
    train_from_file(args.file, som, pause=not args.nopause)

