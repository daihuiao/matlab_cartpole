# matlab_cartpole

This environment uses PPO to train the cartpole from drooping to swinging up. It is a pure script and does not use Simulink. However, the PPO algorithm directly uses Matlab tools and cannot find work similar to stable baseline3 (too bad).

The blue one is the reward curve. Basically, every game can be completed. There is a trade-off between exploration and utilization. The entropy coefficient is finally set to 0.05.

本环境使用使用ppo训练cartpole从下垂到上摆，纯脚本，不用simulink，但是PPO算法直接使用matlab 的工具，找不到类似于stable baseline3的工作（太逊了）。

蓝色的是回报曲线，基本每一局都能完成了，探索和利用要权衡，就那个熵的系数，最后定为0.05

[0598af8ae6f4292106b0f87843d20d0a.mp4](result/0598af8ae6f4292106b0f87843d20d0a.mp4)

![313acacd2e32fa44436281daf1b3e72.png](result/313acacd2e32fa44436281daf1b3e72.png)
