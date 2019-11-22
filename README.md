# Collaboration_Competition_Udacity_DRLND_P3

Project 3: done as part of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). The objective of this project is to create a Multi Agent Deep Deterministic Policy Gradient Learning agent that is able to maximize the reward in the [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) based [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) continuous environment.

## Game Environment Details

![Game Environment](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

The environment has two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

**Note:** This project uses a simulator provided by Udacity which is similar but not identical to the `Tennis` environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

## Getting Started

1. Install project dependencies by following the instructions mentioned at [this link](https://github.com/udacity/deep-reinforcement-learning/#dependencies).

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
     
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

3. Place the file in `/data` directory, and unzip the file.

## Instructions

Following are the steps to train your agent:

1. Clone this github repository:
   ```bash
    git clone https://github.com/anubhavshrimal/Collaboration_Competition_Udacity_DRLND_P3.git
    cd Collaboration_Competition_Udacity_DRLND_P3/
   ```
2. Activate the conda environment where you installed the dependencies and open jupyter notebooks. 
   ```bash
    conda activate drlnd
    jupyter notebook
   ```
3. Open `Tennis.ipynb` on your browser and run all the cells of the notebook.

## Files

* `models/checkpoint_actor_*.pth` and `models/checkpoint_critic_*.pth` are the pre-trained model weights for the Agent, which can be used to further train the Agent or to see how the trained agent performs over the environment
* `Tennis.ipynb` is the ipython notebook which trains the Agent in the reacher environment
* `maddpg` folder contains the implementation for the `Agent` and the `actor`, `critic` models.

## Algorithm

The algorithm and hyper-parameter details are mentioned in [Report.md](./Report.md).