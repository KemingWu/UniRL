# Reward

> **Where it fits:** the *reward* step of the loop —
> rollout → **reward** → advantage → train → sync. In: the rollout engine's
> response `Sample`. Out: per-sample rewards (which the trainer turns into advantages).
> Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/reward-flow-new.png" alt="UniRL reward: RewardService.score_and_attach turns the rollout output (decoded image or text) into a per-sample reward via exactly one backend — local is a single in-process scorer (PickScore, CLIP, HPS, OCR, GenEval2, …) while remote is an HTTP server that runs a panel of reward models (required_rewards) and weight-aggregates them (weighted_sum, mean, min, max) into one reward plus the per-model breakdown; the trainer then z-scores that reward into the advantage" width="100%">
</div>

*One `RewardService` wraps **one backend**: a single in-process scorer (local) or a remote HTTP server that runs and weight-aggregates a **panel** of reward models. The per-sample reward it attaches is what the trainer z-scores into the advantage.*

## What it is

`unirl.reward` scores what a rollout produced — an image, a video, or text — and
writes a per-sample reward back onto the Sample's frontier Part. A `RewardService` wraps
exactly **one** `RewardBackend`: either a local in-process scorer (PickScore, HPS,
CLIP, OCR, GenEval2, math, multiple-choice, video, …) or `RemoteRewardBackend`, a
thin HTTP client for the standalone server in `unirl-reward-service/`.

Turning rewards into advantages is the trainer's job
(`Part.compute_advantages`); generating the media is the rollout engine's.

## Why it exists

RL is only as good as its reward signal, and two things about that signal are
easy to get wrong — so this module owns both:

- **One interface over many scorers.** Local or remote, image/video/text, the
  trainer always calls the same `score_and_attach(sample)` and never touches a
  backend. Swapping PickScore for a remote multi-reward server is a recipe change,
  not a code change.
- **A bad reward must stop the step, not poison it.** A single NaN, null, or
  failed inference call would silently skew a whole GRPO group's advantages. The
  service fails loud: any non-finite or missing reward raises instead of flowing
  into training.

## How it works

Everything goes through one method, `RewardService.score_and_attach(sample)`
(`service.py`). It runs per DP shard (sharding the Sample by prompt-tree), so it
never mutates the input Sample — it returns a fresh one. Per call it:

1. **Refuses precomputed rewards** — raises if the frontier Part already has
   `rewards` (actor-side scoring is the only writer).
2. **Pairs input with output** — the conditioning (`Sample.conditioning`) with the
   media in the frontier Part's `primitive`, already row-aligned (no expansion).
3. **Scores** — hands a typed `RewardRequest` to `backend.compute_rewards`, getting
   back rewards, per-component rewards, and per-sample success flags.
4. **Fails fast** — raises and names the sample if any failed.
5. **Zeroes runaway AR traces** — when the scored Part is itself an AR generation,
   one that hit `max_new_tokens` (never terminated) gets reward 0, so training
   doesn't learn to ramble to the cap.
6. **Attaches** `rewards` + `component_rewards` and returns the Sample.

A backend is just `compute_rewards(request) -> RewardResponse`. Local scorers
(`local/`) subclass `LocalRewardBackend` and implement `_compute_model_rewards`;
the remote backend (`remote.py`) sends one or more bounded `POST /score` calls,
multiplexes every requested reward in each item, and derives success from the
merged response.

### Managed image scorers

`ManagedScorerProcessBackend` is the environment-isolated, rank-affine middle
ground between local and externally deployed rewards. Each reward worker launches
one scorer child with an explicit Python executable; the child inherits that
worker's single visible GPU and serves only its local prompt-tree shard over
loopback HTTP. The initial capability is deliberately limited to image and
image-edit histories.

Its config separates process ownership, scorer construction, and remote-client
semantics:

```yaml
backend:
  _target_: unirl.reward.managed_process.ManagedScorerProcessBackend
  base_device: cpu
  config:
    _target_: unirl.reward.managed_process.ManagedScorerProcessSpec
    process:
      _target_: unirl.reward.managed_process.ManagedProcessConfig
      python_executable: /venvs/reward/bin/python
      service_root: /workspace/UniRL/unirl-reward-service
    scorer:
      _target_: unirl.reward.managed_process.ManagedScorerConfig
      name: editreward
      history_kind: image_edit
      params: {device: cuda, checkpoint_path: /models/EditReward}
    client:
      _target_: unirl.reward.remote.RemoteRewardSpec
      base_url: managed://rank-affine
      required_rewards: [editreward]
      input_kind: image
      request_batch_size: 8
    gpu_residency: resident
```

`request_batch_size` bounds transport/scorer calls independently from the DP
shard size. Identity echo is required for managed children. The parent manages
GPU residency through the child's `onload`, `offload`, and `shutdown` endpoints;
`drain` remains available for explicit synchronization.

**Extending it:** a new local scorer is usually a file in `local/` subclassing
`LocalRewardBackend` (set `canonical_model_name`, implement `_load_model` +
`_compute_model_rewards`, add a `<Name>Spec`), wired in a recipe by `_target_`. A
new remote reward needs no UniRL code — add it to the server and list its name in
`RemoteRewardSpec.required_rewards`.

## Generative-judge rewards (`_judges/`)

`_judges/` holds one module per judge model: the instruction text that judge was
trained on, plus the parser for its free-text reply. `OpenAIChatRewardBackend`
(`remote.py`) uses them to talk straight to a fleet of vLLM
`/v1/chat/completions` servers and score **client-side** — unlike
`RemoteRewardBackend`, which delegates scoring to a RewardService server.

`unirl-reward-service/reward_service/scorers/text_rendering_judge.py` keeps its
own verbatim copy of the same contract so the two packages stay independent
Python distributions. Change the judge's prompt format and both must be synced.

### Gotchas

- **`max_new_tokens` must be 8192, not 4096.** An error-dense infographic needs
  4000+ tokens; 4096 truncated ~1% of images mid-JSON. The judge's real context
  is 16384.
- **A truncated reply can still parse.** `parse_errors`' third tier recovers the
  last complete `{...}` before the cut, so a truncated answer scores as if it
  were whole and every error after the cut vanishes with no trace in the dump.
  Checking `finish_reason == "length"` is mandatory, not optional.
- **`error_severity` returns `None`, never 0, when undefined.** It is defined only
  for 内容正确性与完整性错误; layout/style errors compare a *relation*, not a
  spelling. Callers must charge `None` a separate cost — a reported error that
  scores 0 is invisible to GRPO.
- **Severity gates on the declared category, not on "quotes on both sides."**
  Layout/style descriptions do quote text on the requirement side
  (原始要求【「Oil」位于表头行】/ 实际呈现【实际位于左侧列】) with nothing quoted on the
  actual side. Reading that as "text is missing" scored a perfectly-rendered but
  misplaced label at maximal severity, and left `undefined_severity_cost` dead.
- **The 实际呈现 side is read by intent, not by first quoted run.** When text is
  missing or garbled the judge names the string on *both* sides
  (原始要求【出现文字「Meals」】/ 实际呈现【画面中未出现文字「Meals」】); differencing those
  two quotes yields 0, which made the worst failure free while a one-letter typo
  cost 0.2 — the gradient pointed the wrong way, on 48.6% of all errors in a
  122k-sample dump. Order is 呈现为「X」 (grade it) → absent/garbled (maximal),
  since one description can carry both markers.
- **`weighted_error_cost` changes the reward scale, so rescale `alpha`.**
  Severity ≤ 1 makes the cost strictly ≤ `len(errors)` — ~0.93x on a 122k-sample
  klein-9B dump, since most content errors are absent-or-garbled (severity 1.0)
  and only ~13% land strictly between 0 and 1.
- **`garbled_cost` exists because equal costs reward drawing nothing.** With
  "unreadable" and "never appeared" both at 1.0, omitting text is a safe way to
  dodge a garble error: on a 263k-sample dump of `klein9b_severity_only_fixed`
  the top reward quartile carried *more* fully-unrendered text than the bottom
  (+2.9pp ± 1.4). At `garbled_cost=0.85` that flips to -15.9pp ± 1.4. `None`
  (the default) keeps them equal, i.e. exact V3 behaviour.
- **`max_failure_ratio` neutralizes, it does not zero.** Tolerated failures are
  filled with the mean of the *succeeded* rewards, so their group-relative
  advantage is ~0. Filling 0.0 would turn one random judge parse hiccup into a
  strong "this image was terrible" signal.

## Gotchas

- **A non-finite/missing reward fails the whole step, by design** — fix the scorer.
  `raise_on_failure=False` (remote only) does *not* let training continue on it: the
  backend returns zeros with `successes=[False]`, and `score_and_attach`'s fail-fast
  then raises on those flags anyway. So it can't silently zero-poison a group; leave
  it `True`.
- **`input_kind` must match the media** (`image`/`video`/`text`) — it picks which
  decoded key the backend sees. Remote allows only `image`/`video`; local scorers
  may be `text`.
- **`base_device` is ignored by the remote backend** (it's HTTP-only); local
  scorers honor it, falling back to CPU with a warning if CUDA is unavailable.
