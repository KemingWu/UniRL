"""Wan action-conditioned video model as an embodied environment.

Uses a pretrained Wan DiT + VAE to generate future video frames from
current observations and action sequences. A reward classifier scores
the generated frames to produce per-step rewards.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import DictConfig
from PIL import Image

from unirl.embodied.envs.base import BaseEmbodiedEnv


class WanWorldModelEnv(BaseEmbodiedEnv):
    """Wan video generation model wrapped as a gym-like environment.

    Config fields:
        model_path: path to Wan DiT checkpoint
        vae_path: path to Wan VAE checkpoint
        reward_model.type: "ResnetRewModel" or "TaskEmbedResnetRewModel"
        reward_model.from_pretrained: path to reward classifier
        initial_image_path: path to episode dataset (npy trajectories)
        chunk: action chunk size (default 8)
        condition_frame_length: reference frames for conditioning (default 5)
        num_frames: total frames = condition + chunk (default 13)
        num_inference_steps: diffusion steps (default 5)
        image_size: [H, W] (default [256, 256])
        task_suite_name: e.g. "libero_spatial"
        max_episode_steps: episode horizon
        reward_coef: reward scaling (default 5.0)
        success_reward_threshold: threshold for success (default 0.9)
        enable_kir: keyframe-interpolation-rollout trick (default True)
        group_size: episodes per initial state (default 1)
    """

    def __init__(self, cfg: DictConfig, num_envs: int, device: torch.device, **kwargs):
        super().__init__(cfg, num_envs, device, **kwargs)

        self._chunk = int(cfg.get("chunk", 8))
        self._condition_frame_length = int(cfg.get("condition_frame_length", 5))
        self._num_frames = int(cfg.get("num_frames", 13))
        assert self._num_frames == self._condition_frame_length + self._chunk
        self._image_size = tuple(cfg.get("image_size", [256, 256]))
        self._action_dim = int(cfg.get("action_dim", 7))
        self._num_inference_steps = int(cfg.get("num_inference_steps", 5))
        self._enable_kir = bool(cfg.get("enable_kir", True))
        self._reward_coef = float(cfg.get("reward_coef", 5.0))
        self._success_threshold = float(cfg.get("success_reward_threshold", 0.9))
        self._retain_action = bool(cfg.get("retain_action", True))

        self._group_size = int(cfg.get("group_size", 1))
        self._elapsed_steps = 0

        # State buffers
        self._current_obs: Optional[torch.Tensor] = None  # [B, C, 1, T, H, W]
        self._image_queue: List[List[Optional[torch.Tensor]]] = [
            [None] * self._condition_frame_length for _ in range(num_envs)
        ]
        self._condition_action = torch.zeros(num_envs, self._condition_frame_length, self._action_dim)
        self._task_descriptions: List[str] = [""] * num_envs
        self._prev_step_reward = torch.zeros(num_envs, device=device)

        self._trans_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Lazy-load models
        self._pipe = None
        self._reward_model = None
        self._dataset = None
        self._is_offloaded = False

    def _ensure_loaded(self):
        if self._pipe is None:
            self._pipe = self._build_pipeline()
            self._reward_model = self._load_reward_model()
            self._dataset = self._build_dataset()

    def _build_pipeline(self):
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        device_str = str(self._device)
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device_str,
            model_configs=[
                ModelConfig(path=self._cfg.model_path, offload_device="cpu"),
                ModelConfig(path=self._cfg.vae_path, offload_device="cpu"),
            ],
        )
        pipe.dit.to(self._device)
        pipe.vae.to(self._device)
        return pipe

    def _load_reward_model(self):
        rm_cfg = self._cfg.reward_model
        if rm_cfg.type == "ResnetRewModel":
            from diffsynth.models.reward_model import ResnetRewModel

            model = ResnetRewModel(rm_cfg.from_pretrained)
        elif rm_cfg.type == "TaskEmbedResnetRewModel":
            from diffsynth.models.reward_model import TaskEmbedResnetRewModel

            model = TaskEmbedResnetRewModel(
                checkpoint_path=rm_cfg.from_pretrained,
                task_suite_name=self._cfg.task_suite_name,
            )
        else:
            raise ValueError(f"Unknown reward model type: {rm_cfg.type}")
        return model.eval().to(self._device)

    def _build_dataset(self):
        from unirl.embodied.envs._trajectory_dataset import NpyTrajectoryDataset

        return NpyTrajectoryDataset(self._cfg.initial_image_path, enable_kir=self._enable_kir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._chunk

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @torch.no_grad()
    def reset(
        self,
        *,
        episode_indices: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_loaded()
        self.onload()
        self._elapsed_steps = 0
        self._prev_step_reward.zero_()

        if episode_indices is None:
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(seed)
            episode_indices = torch.randint(0, len(self._dataset), (self._num_envs,), generator=g)

        if isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        img_tensors = []
        task_descriptions = []
        condition_actions = []

        for env_idx, ep_idx in enumerate(episode_indices):
            episode_data = self._dataset[int(ep_idx)]
            first_frame = episode_data["start_items"][0]
            task_descriptions.append(str(episode_data.get("task", "")))

            img_tensor = first_frame["image"]  # [C, H, W] float [0,1]
            if img_tensor.shape[1:] != self._image_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0), size=self._image_size, mode="bilinear", align_corners=False
                ).squeeze(0)
            img_tensor = self._trans_norm(img_tensor)

            env_img_tensor = img_tensor.unsqueeze(1).repeat(1, self._condition_frame_length, 1, 1)
            env_cond_action = np.zeros((self._condition_frame_length, self._action_dim), dtype=np.float32)

            # KIR: use target frames as condition frames
            target_items = episode_data.get("target_items", [])
            if self._enable_kir and len(target_items) == self._condition_frame_length - 1:
                for t_idx, target_frame in enumerate(target_items):
                    t_img = target_frame["image"]
                    if t_img.shape[1:] != self._image_size:
                        t_img = F.interpolate(
                            t_img.unsqueeze(0), size=self._image_size, mode="bilinear", align_corners=False
                        ).squeeze(0)
                    t_img = self._trans_norm(t_img)
                    env_img_tensor[:, t_idx + 1] = t_img
                    env_cond_action[t_idx + 1] = target_frame["action"]

            img_tensors.append(env_img_tensor)
            condition_actions.append(torch.from_numpy(env_cond_action))

        stacked = torch.stack(img_tensors, dim=0).to(self._device)
        self._current_obs = stacked.unsqueeze(2)  # [B, C, 1, T, H, W]
        self._condition_action = torch.stack(condition_actions, dim=0).to(self._device)
        self._task_descriptions = task_descriptions

        for env_idx in range(self._num_envs):
            self._image_queue[env_idx] = [
                self._current_obs[env_idx, :, 0, t : t + 1, :, :] for t in range(self._condition_frame_length)
            ]

        obs = self._wrap_obs()
        return obs, {}

    @torch.no_grad()
    def chunk_step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self.onload()
        autocast_ctx = (
            torch.amp.autocast(device_type=self._device.type, dtype=torch.bfloat16)
            if self._device.type != "cpu"
            else nullcontext()
        )
        with autocast_ctx:
            self._infer_next_chunk_frames(actions)

        self._elapsed_steps += self._chunk
        obs = self._wrap_obs()
        chunk_rewards = self._infer_next_chunk_rewards()

        # Termination from reward threshold
        max_reward = chunk_rewards.max(dim=1)[0]
        terminations = torch.zeros(self._num_envs, self._chunk, dtype=torch.bool, device=self._device)
        terminations[:, -1] = max_reward >= self._success_threshold

        truncations = torch.zeros_like(terminations)
        if self._elapsed_steps >= self.max_episode_steps:
            truncations[:, -1] = True

        infos: Dict[str, Any] = {}
        return obs, chunk_rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer_next_chunk_frames(self, actions: torch.Tensor):
        actions_tensor = actions.to(self._device)
        self._condition_action = self._condition_action.to(device=actions_tensor.device, dtype=actions_tensor.dtype)

        if self._retain_action:
            actions_tensor = torch.cat([self._condition_action, actions_tensor], dim=1)

        self._condition_action[:, 1:, :] = actions_tensor[:, -(self._condition_frame_length - 1) :, :]

        batch_input_image = []
        batch_input_image4 = []
        for env_idx in range(self._num_envs):
            imgs = []
            for frame in self._image_queue[env_idx]:
                frame_np = frame[:, 0].cpu().numpy()  # [C, H, W]
                img = np.transpose(frame_np, (1, 2, 0))
                if img.max() <= 1.2:
                    img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255)
                imgs.append(Image.fromarray(img.astype(np.uint8)))
            batch_input_image.append(imgs[0])
            batch_input_image4.append(imgs[-4:])

        output = self._pipe(
            seed=0,
            tiled=False,
            input_image=batch_input_image,
            input_image4=batch_input_image4,
            action=actions_tensor,
            height=self._image_size[0],
            width=self._image_size[1],
            num_frames=self._num_frames,
            num_inference_steps=self._num_inference_steps,
            cfg_scale=1.0,
            progress_bar_cmd=lambda x: x,
            batch_size=self._num_envs,
        )

        all_samples = []
        for env_idx in range(self._num_envs):
            frames = []
            for img in output[env_idx]:
                arr = np.asarray(img, dtype=np.float32) / 255.0
                arr = arr * 2.0 - 1.0
                frames.append(arr)
            video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # [T, C, H, W]
            video = torch.from_numpy(video).transpose(0, 1)  # [C, T, H, W]

            # Update image queue with last 4 generated frames
            for t in range(video.shape[1] - 4, video.shape[1]):
                self._image_queue[env_idx][t - self._chunk] = video[:, t : t + 1]

            all_samples.append(video[:, self._condition_frame_length :])

        x_samples = torch.stack(all_samples, dim=0).to(self._device, dtype=self._current_obs.dtype)
        x_samples = x_samples.unsqueeze(2)
        self._current_obs = torch.cat([self._current_obs, x_samples], dim=3)

        max_frames = self._condition_frame_length + self._chunk
        if self._current_obs.shape[3] > max_frames:
            self._current_obs = self._current_obs[:, :, :, -max_frames:, :, :]

    def _infer_next_chunk_rewards(self) -> torch.Tensor:
        num_envs, c, v, t, h, w = self._current_obs.shape
        chunk_obs = self._current_obs.permute(0, 3, 1, 2, 4, 5)[:, -self._chunk :, :, :, :, :]
        chunk_obs = chunk_obs.reshape(num_envs * self._chunk, c, v, h, w).squeeze(2)
        chunk_obs = chunk_obs.to(self._device)

        rm_cfg = self._cfg.reward_model
        if rm_cfg.type == "ResnetRewModel":
            rewards = self._reward_model.predict_rew(chunk_obs)
        elif rm_cfg.type == "TaskEmbedResnetRewModel":
            instructions = []
            for env_idx in range(num_envs):
                instructions.extend([self._task_descriptions[env_idx]] * self._chunk)
            rewards = self._reward_model.predict_rew(chunk_obs, instructions)
        else:
            raise ValueError(f"Unknown reward model type: {rm_cfg.type}")

        return rewards.reshape(num_envs, self._chunk)

    def _wrap_obs(self) -> Dict[str, Any]:
        last_frame = self._current_obs[:, :, 0, -1, :, :]  # [B, C, H, W]
        full_image = last_frame.permute(0, 2, 3, 1)  # [B, H, W, C]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image.float(), 0, 255).to(torch.uint8)
        return {
            "main_images": full_image,
            "wrist_images": None,
            "states": torch.zeros(self._num_envs, 16, device=self._device),
            "task_descriptions": self._task_descriptions,
        }

    def offload(self):
        if self._is_offloaded or self._pipe is None:
            return
        self._pipe.dit.to("cpu")
        self._pipe.vae.to("cpu")
        if self._reward_model is not None:
            self._reward_model.to("cpu")
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded or self._pipe is None:
            return
        self._pipe.dit.to(self._device)
        self._pipe.vae.to(self._device)
        if self._reward_model is not None:
            self._reward_model.to(self._device)
        self._is_offloaded = False
