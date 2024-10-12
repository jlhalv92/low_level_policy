## ENVS GO1

* nominal

------------Multi-task Benchmark-----------------------------

* nominal_parkour
* gaps
* hurdles
* steps
* ramps

Example
```
task_name ="ramps"
mdp = Go1(horizon, gamma,
              n_substep=4,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name) ``` 
