## Hybrid Adversarial Policy Learning for Multi-Obstacle Avoidance

# Tasks available
* nominal
  * Environment for Low level Policy. GO1 in open-space that needs to follow joystick commands (Vx, Vy, Wz)
* navigation
  * Not in use, A target position is define and the go1 should safely navigates towards it
 * obstacles_navigation (Training Task)
   * GO1 inside a close room with a single moving cubic obstacle of size 1x1
* multi_obstacles_navigation (Target Task)
  * GO1 inside a close room with three Obstacles. One of the obstacles can move while the others are static
  * The obstacles have different sizes 

# Training Scripts
    * train_adversarial_nav_go1.py
    * train_qarl_nav_go1.py
    (Missing) train_rarl_nav_go1.py (Aryaman's task)

# Conda Env
  Please find the conda enviroment configuration in the respository on the conda_env directory

