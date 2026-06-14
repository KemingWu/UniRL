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
| `ltx2_t2v_dancegrpo_trainside_lora32_33f_10step_0614b_mycode` | SUBMITTED (0614) | LTX2 T2V DanceGRPO trainside. Verifies the audiovisual-forward fix. Prior `0614a` got past the noise-recipe fix but ENDed at transformer forward with `TypeError: forward() missing 2 required positional arguments: 'audio_hidden_states' and 'audio_encoder_hidden_states'` (LTX-2 is a unified audiovisual DiT). Fixed: `predict_noise` rewrite (zeroed audio placeholder + `isolate_modalities=True` + RoPE geometry, discard audio out). Stacked on the noise-recipe fix. |
| `flux2klein_4b_editreward_0613i_mycode` | RUNNING | Flux2Klein 4B + EditReward. New: ckpt save (adapter, every 100), input+output image side-by-side in wandb media. Fixed: data paths (jsonl rewritten to absolute) |
| `hi3_it2i_dancegrpo_editreward_mimo_fsdp_0614b_2node` | SUBMITTED (0614, instance 8b1d89e99eb5dfba019ec60be1842233) | HI3 80B it2i editreward, 2-node (node0=reward, node1=train). Prior `0614a` ENDed at rollout with `AttributeError: 'HunyuanImage3Config' object has no attribute 'model_version'` (ckpt remote `load_tokenizer` reads `self.config.model_version`, never defined). Fixed via `_ensure_tokenizer_loaded` backfill (commit `57e4805`). Also: ENTRY=train_diffusion, 2-node single-train-node pattern (bypasses Ray head IP issue). |

### Key Code Changes in My_Code/UniRL (uncommitted)

1. **unirl/trainer/diffusion.py**: `get_object` instead of `get_class` in `_resolve_noise_latent_shape` (fixes factory-method `_target_` like `LTX2Pipeline.from_bundle`)
2. **unirl/models/ltx2/bundle.py**: `AutoencoderKLLTX2Video` (was wrong `AutoencoderKLLTX2`), `Gemma3ForConditionalGeneration` (was wrong `GemmaForCausalLM`)
3. **unirl/models/ltx2/pipeline.py**: Added `self.shift = config.shift` (engine needs it for FlowMatchSchedulePolicy). Wired the driver x_T noise recipe: added `latent_shape` classmethod (5D `(128, T_lat, H_lat, W_lat)`, 32x spatial/8x temporal), pack/unpack (`_pack_latents`/`_unpack_latents` from diffusers), `_denormalize_latents`, and `_patch_sizes`. `generate` now regenerates x_T via `NoiseRecipe.from_rollout_req(req).resolve()` (raw 5D noise → pack → diffuse → unpack → denormalize → VAE decode) instead of requiring a pre-shipped `request_conditions['initial_latents']`. **Fixes `ValueError: initial_latents must be provided`.** Geometry constants moved to `config.py` (`LTX2_{SPATIAL,TEMPORAL}_COMPRESSION`, `LTX2_LATENT_CHANNELS`).
3b. **unirl/models/ltx2/diffusion.py**: Rewrote `predict_noise` to match the REAL LTX-2 **audiovisual** transformer `forward` — it always runs both video+audio branches and returns `(video_out, audio_out)`. For pure T2V we feed a zeroed 1-frame audio placeholder, set `isolate_modalities=True` (disables a2v/v2a cross-attn so audio can't perturb video), reuse the video text embeds for `audio_encoder_hidden_states`, pass `audio_timestep=timestep`, and thread the video LATENT geometry (`num_frames/height/width` via new `_latent_geometry`) for RoPE coords; discard the audio output. No audio_vae needed (audio dims come off `transformer.config`). Updated both `generate` and `replay` call sites. **Fixes `TypeError: forward() missing 2 required positional arguments: 'audio_hidden_states' and 'audio_encoder_hidden_states'`.**
4. **unirl/models/ltx2/conditions.py**: Added `@dataclass` decorator + `from dataclasses import dataclass` (was missing, breaks Batch serialization)
5. **unirl/types/media_preview.py**: Input+output image side-by-side concat for image-edit tasks (`_hconcat_pil` + `req.primitives["image"]` lookup)
6. **examples/diffusion/flux2_klein/flux2_klein_4b_editreward.yaml**: Added `save_interval/save_mode/save_dir` (ckpt saving, adapter mode, every 100 rollouts)
7. **datasets/image_edit/train.jsonl + test.jsonl**: Rewritten `data/` relative URIs to absolute `/apdcephfs_hldy2/share_305110755/hunyuan/kmwu/datasets/...` (originals backed up as .bak)
8. **unirl/models/hunyuan_image3/bundle.py + text_embed.py**: Added `_ensure_tokenizer_loaded(transformer, tokenizer)` helper — backfills `transformer.config.model_version` (placeholder `"3.0"`) before the ckpt's `load_tokenizer`, which reads `self.config.model_version` but neither the config class nor config.json define it (the value is pass-through-only into the tokenizer's ignored `**kwargs`). Replaces the two raw `load_tokenizer` call sites. **Fixes `AttributeError: 'HunyuanImage3Config' object has no attribute 'model_version'`.**

### Job Scripts (in /apdcephfs_hldy/private_charlesswu/workspace/jobs/reproduce_scripts/jobs/)

| File | Purpose |
|---|---|
| `launch_ltx2_t2v_mycode.sh` | LTX2 trainside, My_Code checkout, installs local diffusers |
| `launch_flux2klein_editreward_2node_mycode.sh` | Flux2Klein 2-node, My_Code checkout |
| `launch_hi3_it2i_editreward_2node_mycode.sh` | HI3 2-node (node0=reward, node1=train_diffusion), My_Code checkout |
| `launch_hi3_it2i_editreward_4node.sh` | HI3 4-node (modified: shared-file head IP broadcast + NODE_IP export + matching runner IP algo). NOT WORKING yet — use 2-node instead |

### Known Issues / Next Steps

- **hi3 4-node**: Ray head IP resolution still broken in practice (works in theory after fix, but never got a clean log to confirm). 2-node bypasses it. If 80B OOMs on single node, will need to revisit.
- **LTX2 dynamic shift**: Currently uses static `shift=1.0` (config default). LTX-2 docs say "dynamic shift based on resolution". May need `build_schedule_policy()` for optimal quality.
- **LTX2 connector**: `from diffusers.pipelines.ltx2.connectors import LTX2Connector` is wrong class name (should be `LTX2ConnectorTransformer1d`), but wrapped in try/except so doesn't crash — connector just won't load.
- **Checkpoints saved to**: `/apdcephfs_hldy/private_charlesswu/workspace/jobs/checkpoints/flux2klein_4b_editreward/`
