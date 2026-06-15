# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Default code agent guidance lives here for Claude Code. Other agents, including Cursor, should reference this file with `@CLAUDE.md` and add only agent-specific differences.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Contribution Hygiene

**Avoid duplicate work, trivial PRs, and unreviewed agent output.**

Before proposing or opening a PR:
- Check the relevant issue, open PRs, and short area keywords for overlapping work.
- If another open PR already addresses the same fix, do not open a duplicate.
- If your approach is materially different from existing work, explain the difference before proceeding.

Do not create PRs for low-value busywork:
- No one-off typo fixes, isolated style churn, or mechanical cleanups without substantive work.
- Bundle mechanical cleanup only when it directly supports a meaningful change.

For AI-assisted work:
- A human submitter must understand and be able to defend every changed line.
- The submitting human should review the full diff and run relevant tests before publication.
- PR descriptions should mention AI assistance, duplicate-work checks, and test commands with results.

Fail closed when the work is not ready:
- If the change is duplicate, too trivial, missing context, or lacks a credible verification path, stop and explain what is missing.
- Do not invent process exceptions just to keep moving.

## 6. Review and Domain Guides

**Verify guidance against the current repo before applying it.**

- Treat agent or bot review comments as suggestions, not facts. Confirm they still apply to the current code before changing anything.
- Before editing specialized areas, read and follow the relevant local guide or skill.
- If a guide conflicts with the requested change, refuse that part of the change and explain the conflict.

Local skills currently in this repo:

| Area | Skill | Read before |
| --- | --- | --- |
| Model bundles | `.claude/skills/development/add-model-bundle/SKILL.md` | Adding or updating diffusion or autoregressive model pipelines, model config dataclasses, Bundle/Pipeline/Stage/Conditions implementations, LoRA targets, FSDP wrapping hints, RolloutReq/RolloutResp plumbing, or multimodal text/image/video conditioning. |
| Pull requests | `.claude/skills/development/pr-workflow/SKILL.md` | Creating or updating PRs, editing PR bodies, handling PR Body or Semantic Pull Request CI failures, or running `gh pr create`. |

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Session State (2026-06-14)

### Running Jobs (check status with `taiji_client trl`)

| Task Flag | Status | What It's Testing |
|---|---|---|
| `ltx2_t2v_dancegrpo_trainside_lora32_33f_10step_0614f_mycode` | SUBMITTED (0614, instance 8b1d81849eb5e039019ecab3361e290a, commit `9d9a413`) | LTX2 T2V DanceGRPO trainside. Verifies the audio co-denoise fix for RESIDUAL blur. Prior `0614e` (dynamic-σ fix) made content visible but videos were still soft — the audio→video cross-attn residual was deleted (1-frame zero audio + isolate_modalities=True). Fixed (change #11): co-denoise a real audio latent stream (isolate_modalities=False), store audio trajectory in `LatentSegment.aux_latents` for replay. Also switched data to `datasets/vid_prompt/` (video prompts). Stacked on σ + all prior fixes. |
| `flux2klein_4b_editreward_0614a_mycode` | SUBMITTED (0614, instance 8b1d810e9eb5c31b019ec969e9d02562, commit `b578de5`) | Flux2Klein 4B + EditReward. Verifies the image-edit conditioning fix. Prior `0613i` RAN but the pipeline was pure T2I — `primitives["image"]` (source) never reached the transformer, so edited images ignored the source. Fixed (change #9): source image now VAE-encoded → packed condition tokens concatenated onto the noise sequence. Also has: ckpt save (adapter, every 100), input+output side-by-side wandb media, absolute data paths. |
| `hi3_it2i_dancegrpo_editreward_mimo_fsdp_0614d_2node` | SUBMITTED (0614, commit `<pending>`) | HI3 80B it2i editreward, 2-node. Prior `0614c` got past the tokenizer-path fix but ENDed at `AttributeError: 'FSDPHunyuanImage3ForCausalMM' object has no attribute '_tkwrapper'` — AND was silently running **t2i** (not it2i), so the source image never entered the model. Two-layer fix (change #12): (A) thread recipe `stage_params.task` → `req.stage_config` (was dropped → always defaulted to t2i); (B) `transformer._tkwrapper`→`transformer.tokenizer` (the ckpt's `load_tokenizer` sets `_tokenizer`, no `_tkwrapper`), `apply_chat_template` kwarg `batch_cond_image_info`→`batch_cond_images`, and wrap the source image into `CondImage(image_type="vae_vit")` objects for the edit conditioning. |

### Key Code Changes in My_Code/UniRL (uncommitted)

1. **unirl/trainer/diffusion.py**: `get_object` instead of `get_class` in `_resolve_noise_latent_shape` (fixes factory-method `_target_` like `LTX2Pipeline.from_bundle`)
2. **unirl/models/ltx2/bundle.py**: `AutoencoderKLLTX2Video` (was wrong `AutoencoderKLLTX2`), `Gemma3ForConditionalGeneration` (was wrong `GemmaForCausalLM`)
3. **unirl/models/ltx2/pipeline.py**: Added `self.shift = config.shift` (engine needs it for FlowMatchSchedulePolicy). Wired the driver x_T noise recipe: added `latent_shape` classmethod (5D `(128, T_lat, H_lat, W_lat)`, 32x spatial/8x temporal), pack/unpack (`_pack_latents`/`_unpack_latents` from diffusers), `_denormalize_latents`, and `_patch_sizes`. `generate` now regenerates x_T via `NoiseRecipe.from_rollout_req(req).resolve()` (raw 5D noise → pack → diffuse → unpack → denormalize → VAE decode) instead of requiring a pre-shipped `request_conditions['initial_latents']`. **Fixes `ValueError: initial_latents must be provided`.** Geometry constants moved to `config.py` (`LTX2_{SPATIAL,TEMPORAL}_COMPRESSION`, `LTX2_LATENT_CHANNELS`).
3b. **unirl/models/ltx2/diffusion.py**: Rewrote `predict_noise` to match the REAL LTX-2 **audiovisual** transformer `forward` — it always runs both video+audio branches and returns `(video_out, audio_out)`. For pure T2V we feed a zeroed 1-frame audio placeholder, set `isolate_modalities=True` (disables a2v/v2a cross-attn so audio can't perturb video), reuse the video text embeds for `audio_encoder_hidden_states`, pass `audio_timestep=timestep`, and thread the video LATENT geometry (`num_frames/height/width` via new `_latent_geometry`) for RoPE coords; discard the audio output. No audio_vae needed (audio dims come off `transformer.config`). **Fixes `TypeError: forward() missing 2 required positional arguments`.** ALSO rewrote `generate`/`replay` to match the framework SDE contract (mirroring WAN21): use `strategy.denoise(noise_pred, sample, ...) -> (prev_sample, log_prob, prev_sample_mean)` (the invented `denoise_with_logp` never existed), store the trajectory SPARSELY into `LatentSegment.{latents,indices}` via `compute_trajectory_positions` (SDE endpoints ∪ {T}), `init_schedule(sigmas)`, and replay reads `sample=latents_at(k)` / `prev_sample=latents_at(k+1)`. Pipeline now reads the final latent via `segment.latents_at(num_inference_steps)` (the bogus `segment.final_latents` field never existed). **Fixes `AttributeError: 'FlowSDEStrategy' object has no attribute 'denoise_with_logp'` and makes GRPO log-prob replay correct.**
4. **unirl/models/ltx2/conditions.py**: Added `@dataclass` decorator + `from dataclasses import dataclass` (was missing, breaks Batch serialization)
3c. **LTX2 systematic audit fixes (one pass to stop serial crashes)** — `pipeline.py`: `RolloutTrack(parent_ids=...)` not `group_ids=` (a read-only @property), `decoded=<Videos>` not `decoded={"video":...}`, track key `"video"`; CFG empty-`""`-negative synthesis when `guidance_scale>1` and no negative. `vae.py`: rewrote `decode` → returns varlen-packed `Videos` via `from_list` (was `Videos(frames=5d)` which skips cu_seqlens), `(x+1)/2` rescale from `[-1,1]` (was bare clamp → black frames), fp32 decode, and pass a zeros `timestep` when `vae.config.timestep_conditioning` (else `None*Parameter` crash). `bundle.py`: connector class is `LTX2TextConnectors` not `LTX2Connector` (the latter doesn't exist → was always silently None), load made mandatory (no try/except swallow). `text_embed.py`: stack ALL Gemma hidden layers `(B,seq,C*49)` + left-padding for the connector (was last-layer only → wrong shape/conditioning); connector call is positional `connectors(packed, mask, padding_side=)` returning 3-tuple `(video,audio,binary_mask)` (was kwarg `hidden_states=` + `.video_features` attrs that don't exist).
5. **unirl/types/media_preview.py**: Input+output image side-by-side concat for image-edit tasks (`_hconcat_pil` + `req.primitives["image"]` lookup)
6. **examples/diffusion/flux2_klein/flux2_klein_4b_editreward.yaml**: Added `save_interval/save_mode/save_dir` (ckpt saving, adapter mode, every 100 rollouts)
7. **datasets/image_edit/train.jsonl + test.jsonl**: Rewritten `data/` relative URIs to absolute `/apdcephfs_hldy2/share_305110755/hunyuan/kmwu/datasets/...` (originals backed up as .bak)
8. **unirl/models/hunyuan_image3/bundle.py + text_embed.py**: Added `_ensure_tokenizer_loaded(transformer, tokenizer_path)` helper for the ckpt's lazy `load_tokenizer`. Two fixes: (a) backfills `transformer.config.model_version` (placeholder `"3.0"`) — the ckpt code reads it but neither the config class nor config.json define it (pass-through-only kwarg). (b) **passes the checkpoint PATH/repo-id, not the tokenizer object** — `load_tokenizer(self, tok)` does `HunyuanImage3TokenizerFast.from_pretrained(tok, ...)`, which needs a path string; passing the loaded tokenizer object made `from_pretrained` treat its `repr()` as a path → `OSError: Can't load tokenizer for 'PreTrainedTokenizerFast(...)'` with a full added-tokens dump. Now passes `pretrained_path`. **Fixes the `model_version` AttributeError AND the subsequent tokenizer-load OSError.**
9. **unirl/models/flux2_klein/{conditions,vae,pipeline,diffusion}.py + __init__.py**: Added the missing **image-edit conditioning path** — the pipeline was pure T2I, so the source image (`primitives["image"]`, role=condition) never reached the transformer and edited outputs ignored it (visible in wandb). Mirrors diffusers `Flux2KleinPipeline` reference path: new `Flux2KleinVAEEncodeStage` (resize→`[-1,1]`→`vae.encode().mode()`→patchify→BN-normalize→pack tokens + time-offset 4-axis RoPE ids); `Flux2KleinConditions` gained `image_latent` + `image_latent_ids` slots (round-trip via from_dict/to_dict so replay sees them too); pipeline reads `primitives["image"]` and attaches; `Flux2KleinDiffusionStep.predict_noise` concatenates condition tokens onto the noise sequence (`cat([noise, cond], dim=1)` + ids), runs the transformer, then slices `noise_pred[:, :noise_len]`. Pure T2I unaffected (slots stay None). |
10. **unirl/models/ltx2/schedule.py (new) + pipeline.py + __init__.py + unirl/sde/runtime.py**: Fixed BLURRY LTX2 videos (reward rose anyway because rollout/replay shared the same wrong σ). Root cause: LTX2 had no `build_schedule_policy`, so the engine fell back to static `shift=1.0` (identity σ schedule) — but LTX2 needs diffusers' **dynamic exponential shift, constant μ=2.05** (diffusers pins `calculate_shift`'s `image_seq_len` to `max_image_seq_len` → μ ≡ `max_shift`) plus `shift_terminal=0.1`. Added `LTX2SchedulePolicy(compute_mu→max_shift)` + `build_ltx2_schedule_policy` (base_shift=0.95, max_shift=2.05, base/max_image_seq_len=1024/4096, time_shift_type=exponential, shift_terminal=0.1 — all verified against the ckpt scheduler_config.json), wired via `LTX2Pipeline.build_schedule_policy`. Threaded an optional `shift_terminal` through `compute_sigma`/`get_sigma_schedule` (default None → no-op for other models).
11. **unirl/models/ltx2/diffusion.py + unirl/types/segments/latent.py + examples/diffusion/ltx2/ltx2_t2v_trainside.yaml**: Fixed RESIDUAL blur after #10 (videos showed content but still soft). Root cause: LTX-2 is a unified audiovisual DiT that injects an audio→video cross-attention residual into the video stream at every layer; diffusers' default T2V path **co-denoises a real audio latent stream** with `isolate_modalities=False`. Our shortcut (1-frame zeroed audio + `isolate_modalities=True`) deleted that trained residual at all 48 layers → distribution shift → blur. Now `predict_noise` co-processes a real audio latent (geometry from fixed fallbacks — 25 latent fps, 8ch·16mel=128 feature dim; no audio_vae needed) with `isolate_modalities=False`, returns `(video_pred, audio_pred)`; `generate` ODE-steps audio in lockstep with the video SDE step and stores the audio trajectory in the new `LatentSegment.aux_latents` (sparse, same indices as video); `replay` feeds `aux_latents_at(k)` so the video log-prob matches rollout. Also passes `sigma`/`audio_sigma` (LTX-2.3 modulation; no-op on 2.0). Recipe now uses `datasets/vid_prompt/` (50k video prompts, copied from Flow-Factory) instead of pickscore image prompts. `LatentSegment.aux_latents` defaults None → other models unaffected. |
12. **unirl/trainer/diffusion.py + unirl/train_diffusion.py + unirl/models/hunyuan_image3/{bundle,text_embed,vit_encode}.py + modes/{it2i,i2t,t2t}.py**: Two-layer HI3 fix. **(A) Task routing**: the recipe's `stage_params: {task: it2i}` was never threaded into `req.stage_config`, so `pipeline.generate` always took the default `task="t2i"` → the it2i mode (source-image conditioning) NEVER ran → edits ignored the source. Added `stage_params` to `DiffusionTrainer.__init__` (stored as `_stage_config`), passed from `train_diffusion.py` (`cfg.get("stage_params")`), and set `RolloutReq.stage_config=dict(self._stage_config)` in `_build_req`. **(B) Tokenizer API drift**: `transformer._tkwrapper` does not exist on the ckpt — `load_tokenizer` sets `self._tokenizer` (via the `.tokenizer` property), and `apply_chat_template` lives there. Fixed all call sites (`_tkwrapper`→`tokenizer`; `_ensure_tokenizer_loaded` guard checks `_tokenizer`; t2t stop-token reads `_tokenizer`). Also the `apply_chat_template` cond-image kwarg is `batch_cond_images` (was `batch_cond_image_info`) and wants `list[CondImage]` — `vit_encode.encode_for_cond_vit` now wraps each preprocess result into `CondImage(image_type="vae_vit", vae_image, vit_image)` (imported from the ckpt's tokenization module) and it2i/i2t pass `vit["cond_images"]`. **Fixes the `_tkwrapper` AttributeError AND makes it2i actually condition on the source image.** |

### Job Scripts (in /apdcephfs_hldy/private_charlesswu/workspace/jobs/reproduce_scripts/jobs/)

| File | Purpose |
|---|---|
| `launch_ltx2_t2v_mycode.sh` | LTX2 trainside, My_Code checkout, installs local diffusers |
| `launch_flux2klein_editreward_2node_mycode.sh` | Flux2Klein 2-node, My_Code checkout |
| `launch_hi3_it2i_editreward_2node_mycode.sh` | HI3 2-node (node0=reward, node1=train_diffusion), My_Code checkout |
| `launch_hi3_it2i_editreward_4node.sh` | HI3 4-node (modified: shared-file head IP broadcast + NODE_IP export + matching runner IP algo). NOT WORKING yet — use 2-node instead |

### Known Issues / Next Steps

- **hi3 4-node**: Ray head IP resolution still broken in practice (works in theory after fix, but never got a clean log to confirm). 2-node bypasses it. If 80B OOMs on single node, will need to revisit.
- **LTX2 dynamic shift**: FIXED. Was using static `shift=1.0` (engine fell back to `from_pretrained` static because no `build_schedule_policy`), which under-resolved the trajectory → blurry videos (reward still rose since rollout/replay shared the same σ). Now `LTX2SchedulePolicy` (schedule.py) gives constant μ=2.05 exponential shift + `shift_terminal=0.1`, matching the checkpoint's scheduler_config.json and diffusers' image_seq_len-pinned `calculate_shift`. See change #10.
- **LTX2 connector**: `from diffusers.pipelines.ltx2.connectors import LTX2Connector` is wrong class name (should be `LTX2ConnectorTransformer1d`), but wrapped in try/except so doesn't crash — connector just won't load.
- **Checkpoints saved to**: `/apdcephfs_hldy/private_charlesswu/workspace/jobs/checkpoints/flux2klein_4b_editreward/`
