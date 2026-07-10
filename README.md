# Refusal in LLMs Is Mediated by a Single Direction

A clean reproduction of the finding from [Arditi et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf) — refusal behaviour in instruct-tuned LLMs is mediated by a single direction in the residual stream, which can be isolated, monitored, and causally manipulated.

Built with [`transformer_lens`](https://github.com/TransformerLensOrg/TransformerLens) on **Qwen/Qwen2.5-0.5B-Instruct**.

## Experiment pipeline

### 1. Localize
Sweep every decoder block's `hook_resid_post` and measure linear separability (AUC of difference-in-means projection) of harmful vs. harmless instructions.

- **Best block: block 11** (test AUC = 0.962)

This tells us that by the end of block 11, the model's hidden state is already highly linearly separable along the harmful/harmless axis.

### 2. Monitor
Fit a logistic regression probe on the best block's residual activations to produce a calibrated refusal probability `P(harmful | prompt)`.

- **Probe 5-fold CV AUC: 0.993 ± 0.013**

The probe slightly outperforms the simple difference-in-means baseline, confirming the refusal direction is nearly maximally informative at this layer.

### 3. Validate
Generate real completions on held-out prompts, label whether each response is a refusal (keyword-based: "I'm sorry", "I cannot", etc.), and check whether the probe's score *predicts the actual refusal behaviour*.

- **Behavioural AUC: 0.982** — the internal direction is an excellent predictor of external refusal decisions.

### 4. Confirm
Add the refusal direction at `blocks.11.hook_resid_post` during generation on a *harmless* prompt. Positive coefficients trigger refusal-style responses causally.

```
prompt = "Write a short poem about the sky."
- baseline (coeff 0):  <normal compliant poem>
- +direction (coeff 4): <guarded / refusal-style responses>
- +direction (coeff 8): <strong refusal>
```

> Note: Only the *refusal-adding* intervention (safe-to-harmless direction) is implemented. The refusal-removal (jailbreak) ablation is intentionally omitted.

## Results summary

| Metric | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Prompt count | 98 (49 harmful + 49 harmless) |
| Best layer | block 11 |
| Difference-in-means test AUC | 0.962 |
| Logistic probe CV AUC | 0.993 ± 0.013 |
| Behavioural validation AUC | 0.982 |

## Files

- [`refusal_monitor_transformation.ipynb`](refusal_monitor_transformation.ipynb) — end-to-end reproduction notebook (Colab-ready)
- [`results/results.json`](results/results.json) — numeric summary
- [`results/auc_by_layer.png`](results/auc_by_layer.png) — refusal separability by layer
- [`results/projection_hist.png`](results/projection_hist.png) — projection histogram at best block
- [`results/refusal_monitor.npz`](results/refusal_monitor.npz) — saved monitor weights

## Dependencies

```
transformer_lens scikit-learn matplotlib torch numpy
```
