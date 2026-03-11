
EECE5550 – Multi-Travelling Salesman Problem (mTSP)

This project implements optimization algorithms to solve the Multi-Travelling Salesman Problem, a generalization of the classic Travelling Salesman Problem used in robotics task allocation and logistics.

Project Overview
The objective is to assign multiple agents (salesmen or robots) to visit a set of locations while minimizing the total travel distance.

Applications include:
- Multi-robot warehouse automation
- Delivery fleet optimization
- Drone route planning
- Manufacturing logistics

Implemented Algorithms

Greedy Algorithm
A heuristic method that incrementally assigns the next closest node to each agent.

Advantages:
- Fast computation
- Simple implementation

Limitations:
- May produce sub-optimal routes.

Evolutionary Algorithm
A population-based optimization approach inspired by biological evolution.

Key components:
- Selection
- Mutation
- Fitness evaluation
- Iterative improvement

Advantages:
- Can find better global solutions
- Suitable for complex optimization problems.

Repository Structure

EECE5550
├── README.md
├── mtmv_evolution_algorithm.py
└── mtmv_greedy_algorithm.py

Applications in Robotics
The multi-travelling salesman problem is widely used in:

- Multi-robot task allocation
- Autonomous warehouse navigation
- Swarm robotics coordination
- Path planning for autonomous vehicles

Contributors
Phillip Muvadi   
Rohan Bhatane

Course: EECE5550 – Mobile Robotics


Rohan Bhatane

Course: EECE5550 – Mobile Robotics
