# Implementation of Reinforcement Learning Algorithms in Keras tested on VizDoom

This repo includes implementation of Double Deep Q Network (DDQN), Dueling DDQN, Deep Recurrent Q Network (DRQN) with LSTM, REINFORCE, Advantage Actor Critic (A2C), A2C with LSTM, and C51 DDQN (Distribution Bellman). All implementations are tested on [VizDoom](http://vizdoom.cs.put.edu.pl/) **Defend the Center** scenario, which is a 3D partially observable environment. 

For more details on the implementation, you can check out my blog post at https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html.

<img src="/resources/a2c.gif" width="300">


## Results

Below is the performance chart of 20,000 episodes of **DDQN**, **REINFORCE**, and **A2C** running on Defend the Center. Y-axis is the average number of kills (moving average over 50 episodes).

![Performance Chart 1](/resources/chart_1.png)

The performance chart of 15,000 episodes **C51 DDQN** and **DDQN** running on Defend the Center.

![Performance Chart 2](/resources/chart_2.png)

## Usage

First follow [this](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md) instruction to install VizDoom. If you use python, you can simply do pip install:

```
$ pip install vizdoom
```

Second, clone [ViZDoom](https://github.com/mwydmuch/ViZDoom) to your machine, copy the python files provided in this repo over to `examples/python`. 

To test if the environment is working, run

```
$ cd examples/python
$ python ddqn.py
```

You should see some printouts indicating that the DDQN is running successfully. Errors would be thrown otherwise. 

## Dependencies

* Keras 1.2.2 / 2.0.5
* Tensorflow 0.12.0 / 1.2.1
* [VizDoom Environment](http://vizdoom.cs.put.edu.pl/)





 




