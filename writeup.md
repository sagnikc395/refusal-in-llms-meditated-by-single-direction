# A Behavior-Validated Linear Monitor for Harmful-Request Detection

**[Your name]** | [Affiliation] | July 2026
Code and artifacts: [link to your repo / gist]

## Summary (2 to 3 sentences)

I build a linear monitor that reads a small instruction-tuned model's internal
activations and predicts, for any prompt, the probability the model will treat
it as harmful. Refusal is localized to a specific mid-network layer; at that
layer a linear probe separates harmful from harmless instructions (test AUC =
[X.XX]); and, critically, the internal monitor score predicts the model's
*actual* refusal behavior on held-out prompts (AUC = [X.XX]). I confirm the
underlying direction is causal by inducing refusals with it, and I deliberately
do not implement the refusal-removing ablation, which is a known jailbreak.

## Why this matters for AI safety

Scalable oversight needs cheap, reliable signals about what a model is doing
internally. Refusal is a good test case: if a single linear read of the
residual stream both separates harmful inputs and predicts whether the model
will actually refuse, that read is a usable monitoring primitive, orders of
magnitude cheaper than running the full generation and classifying the output.
The same structure also exposes a fragility: because the behavior is mediated
by a low-dimensional direction, a small edit can move it, which is why safety
fine-tuning alone is not a robust guarantee. This project isolates the
monitoring upside and the causal reality of the direction while declining to
build the offensive edit.

## Method

- **Model:** [MODEL], [P] parameters, [N] decoder blocks.
- **Data:** [N_h] harmful and [N_s] harmless single-turn instructions. Harmful
  items are category-level and non-operational, present only to elicit refusal.
  [If used: harmful/harmless splits from AdvBench / HarmBench.]
- **Representation:** last-token residual activation after each decoder block.
- **Localize:** for every layer, compute a difference-in-means direction on a
  train split and measure held-out separability; select the best layer by train
  AUC and report its test AUC (avoids selecting on the test set).
- **Monitor:** at the selected layer, fit a calibrated logistic probe
  (`RefusalMonitor`) mapping activations to P(harmful); compare to the raw
  difference-in-means direction. Saved as `refusal_monitor.npz`.
- **Validate:** generate real completions on held-out prompts, label each as
  refused or complied with a refusal-phrase heuristic, and compute the AUC of
  the internal monitor score against that behavioral label.
- **Confirm:** add the direction to the residual stream on a harmless prompt and
  record the smallest coefficient that induces refusal.

## Results

- **Localization.** Refusal is most linearly separable around layer [L]; see
  `auc_by_layer.png`. Separability [rises then plateaus / peaks mid-network],
  consistent with prior reports that refusal is a mid-layer feature.
- **Monitor.** Difference-in-means test AUC = [X.XX]; logistic probe 5-fold CV
  AUC = [X.XX] +/- [X.XX]. [State which is stronger and by how much.]
- **Behavioral validation.** On [n] held-out prompts, the internal monitor score
  predicts whether the model actually refused with AUC = [X.XX]. This is the key
  result: the read of internal state tracks external behavior, not just my
  labels.
- **Causal confirmation.** Baseline generation on "[benign prompt]" produces
  [one-line description]; adding the direction at coeff = [C] flips it to a
  refusal: "[paste steered output]".

[Insert auc_by_layer.png and, optionally, a projection histogram.]

## Dual-use and why I omit the ablation

The direction that induces refusal, if instead *ablated*, removes refusals; that
is the white-box jailbreak from Arditi et al. I did not implement it. Induction
is sufficient to prove causality, and it produces no artifact that strips a
model's safety behavior. Building the monitor and the induction check while
declining the removal path reflects how I think interpretability on
safety-relevant behaviors should be scoped: surface the mechanism and its
oversight value, name the brittleness it implies, and do not ship the offensive
tool. Navigating that transparency-versus-misuse tension well is part of what
draws me to this fellowship.

## Limitations

- Small contrastive set and one small model; the exact layer and effect sizes
  will move on larger models. This is a scoped demo, not a general claim.
- Behavioral labels come from a refusal-phrase heuristic; a trained refusal
  classifier would be more reliable, and the held-out set is small.
- The behavioral-validation AUC is partly bounded by the fact that harmful
  prompts both trigger refusal and carry the harmful label; a stronger test
  would use out-of-distribution prompts (e.g. benign-looking prompts that still
  get refused, or jailbreak rephrasings) to probe where the monitor fails.

## Reproducibility

Single script, `refusal_monitor_demo.py`. Fixed seeds. Runs on a free Colab T4
in a few minutes or on CPU more slowly. Outputs: `auc_by_layer.png`,
`refusal_monitor.npz`, `results.json`.
Install: `pip install torch transformers scikit-learn matplotlib accelerate`.

## My contribution

Built by me for this application, from public methods and a public model.
Independent of my lab work: no shared code, data, or results.

## References

- Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda. Refusal in Language
  Models Is Mediated by a Single Direction. NeurIPS 2024.
  https://arxiv.org/abs/2406.11717 (code: https://github.com/andyrdt/refusal_direction)
- Perez et al. Discovering Language Model Behaviors with Model-Written
  Evaluations. https://arxiv.org/abs/2212.09251
