"""Trajectory dataset loader for world-model environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


class NpyTrajectoryDataset:
    """Loads episode trajectories from numpy files for environment initialization.

    Expected directory structure::
        root/
          episode_0/
            images.npy       # [T, H, W, C] uint8
            actions.npy      # [T, action_dim] float32
            task.txt         # single-line task description
          episode_1/
          ...

    Or a single combined .npz file with keys per episode.
    """

    def __init__(self, root_path: str, enable_kir: bool = True):
        self._root = Path(root_path)
        self._enable_kir = enable_kir
        self._episodes = self._discover_episodes()

    def _discover_episodes(self) -> List[Path]:
        root = self._root
        if root.is_file() and root.suffix == ".npz":
            data = np.load(str(root), allow_pickle=True)
            self._npz_data = data
            return list(range(len(data.files) // 3))  # images/actions/task per episode
        episodes = sorted([d for d in root.iterdir() if d.is_dir()])
        if not episodes:
            npy_files = sorted(root.glob("*.npy"))
            if npy_files:
                return npy_files
        return episodes

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_path = self._episodes[idx]

        if isinstance(ep_path, int):
            return self._load_from_npz(ep_path)

        if ep_path.is_file() and ep_path.suffix == ".npy":
            return self._load_single_npy(ep_path)

        return self._load_from_directory(ep_path)

    def _load_from_directory(self, ep_dir: Path) -> Dict[str, Any]:
        images = np.load(str(ep_dir / "images.npy"))  # [T, H, W, C]
        actions = np.load(str(ep_dir / "actions.npy"))  # [T, action_dim]

        task_file = ep_dir / "task.txt"
        task = task_file.read_text().strip() if task_file.exists() else ""

        first_img = torch.from_numpy(images[0]).permute(2, 0, 1).float() / 255.0

        start_items = [{"image": first_img}]
        target_items = []

        if self._enable_kir and len(images) > self._condition_frame_length:
            # Use evenly spaced frames as KIR targets
            n_targets = 4  # condition_frame_length - 1
            indices = np.linspace(1, len(images) - 1, n_targets, dtype=int)
            for i in indices:
                t_img = torch.from_numpy(images[i]).permute(2, 0, 1).float() / 255.0
                t_action = actions[min(i, len(actions) - 1)]
                target_items.append({"image": t_img, "action": t_action})

        return {"start_items": start_items, "target_items": target_items, "task": task}

    def _load_from_npz(self, idx: int) -> Dict[str, Any]:
        data = self._npz_data
        images = data[f"images_{idx}"]
        actions = data[f"actions_{idx}"]
        task = str(data.get(f"task_{idx}", ""))

        first_img = torch.from_numpy(images[0]).permute(2, 0, 1).float() / 255.0
        start_items = [{"image": first_img}]
        return {"start_items": start_items, "target_items": [], "task": task}

    def _load_single_npy(self, path: Path) -> Dict[str, Any]:
        data = np.load(str(path), allow_pickle=True).item()
        if isinstance(data, dict):
            images = data.get("images", data.get("obs", np.zeros((1, 256, 256, 3), dtype=np.uint8)))
            first_img = torch.from_numpy(images[0]).permute(2, 0, 1).float() / 255.0
            task = data.get("task", "")
            return {"start_items": [{"image": first_img}], "target_items": [], "task": task}
        first_img = torch.from_numpy(data[0]).permute(2, 0, 1).float() / 255.0
        return {"start_items": [{"image": first_img}], "target_items": [], "task": ""}

    @property
    def _condition_frame_length(self) -> int:
        return 5
