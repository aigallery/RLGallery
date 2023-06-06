# RL Gallery

## Algorithms

> Note: All the base code is from "Hands on Reinfocement Learning".


- Off-Policy
    - VanillaDQN
    - DoubleDQN
    - DuelingDQN
    - DDPG
    - SACContinuous
- On-Policy
    - VPG(REINFORCE)
    - ActorCritic
    - TRPO
    - PPO
    - PPOContinuous


## Registry

It allows for 

- using decrator to register a new class, and the self.xxx can be overwritten by config.yaml
- building an intance by Registry().build({"type":"className"}), the dict is the key-value pair for the __init__() of the given class. "type" is fixed for the className's key.
- since we set all the default values in the class's __init__(),  so Registry().build({"type":"className"}) can write the least number of parameter.
- if you want to convey param by {"type":"className", "param1":"value1"} this way, remember to comment the coresponding items in the config.yaml, since the value can be overwritten by this file.

## Config.yaml

Every class can set param in this file, the file's priority is the highest.

## Wandb

The log system.

## Runner

All the supported algorithm has been registried in this file.

## Main

The Gallery's enter point.



![logs](https://cdn.staticaly.com/gh/HongdaChen/image-home@master/20230606/image.37oy1dy4s0c0.png)


