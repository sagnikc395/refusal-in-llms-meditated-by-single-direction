## refusal-in-lms-meditated-by-single-direction

an re-implementation of the paper : https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf

### pipeline of execution:

1. Localize -> sweep every block's internal residual stream, measure linear seperability of harmful vs harmless instructions, then selects the best layer 
2. Monitor  -> fit a caliberated logistic probe on that layer's activations (prompt -> P(harmful)) and then compare to difference-in-means and saves the result
3. Validate  -> generate the real completions on held-out prompts , label the refusals and show the internal score predicts the external behaviour
4. Confirm ->  add the direction at hook_resid_post to induce refusal on a harmless prompt (safe direction only; the refusal-REMOVING ablation, i.e. the jailbreak, is intentionally omitted).

