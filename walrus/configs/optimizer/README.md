## Key settings overview

Optimizer settings

```yaml
_target_: torch.optim.AdamW # Standard torch optimizer
weight_decay: 1E-4 
eps: 1e-10 # Lower epsilon out of paranoia about small gradients
lr: 2e-4

param_groups: # Optional - can be used to assign different LRs to different layers. Not used in paper.
  - params: embed
    lr: 5e-5
  - params: debed
    lr: 5e-5
  - params: blocks
    lr: 1e-6
  - params: encoder_dummy
```