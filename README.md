# Help Me Explore: Minimal Social Interventions for Graph-Based Autotelic Agents
_This repository contains the code associated to the Help Me Explore: Minimal Social Interventions for Graph-Based Autotelic Agents paper submitted at the ICML 2022 conference._

**Abstract.**
In the quest for autonomous agents learning open-ended repertoires of skills, most works take a Piagetian perspective: learning trajectories are the results of 
interactions between developmental agents and their physical environment. The Vygotskian perspective, on the other hand, emphasizes the centrality of the 
socio-cultural environment: higher cognitive functions emerge from transmissions of socio-cultural processes internalized by the agent. This paper acknowledges
these two perspectives and presents GANGSTR, a hybrid agent engaging in both individual and social goal-directed exploration. In a 5-block manipulation domain, 
GANGSTR discovers and learns to master tens of thousands of configurations. In individual phases, the agent engages in autotelic learning; it generates, pursues
and makes progress towards its own goals. To this end, it builds a graph to represent the network of discovered configuration and to navigate between them. 
In social phases, a simulated social partner suggests goal configurations at the frontier of the agent’s current capabilities. This paper makes two 
contributions: 1) a minimal social interaction protocol called Help Me Explore (HME); 2) GANGSTR, a graph-based autotelic agent. As this paper shows, coupling
individual and social exploration enables the GANGSTR agent to discover and master the most complex configurations (e.g. stacks of 5 blocks) with only minimal
intervention.

**Link to Website**

Link to our website will be available soon with additional illustrations and videos.

**Requirements**

* gym
* mujoco
* pytorch
* pandas
* matplotlib
* numpy
* networkit

To reproduce the results, you need a machine with **24** cpus.

**Simulated Social Partner**

The _HME_ interaction protocol relies on a simulated social partner that possesses a model of the agent's learned and learnable skills. 
We represent this knowledge with a semantic graph of connected configurations. This graph is already generated and available under 
the _graph/_ folder.

**Training GANGSTR**

| Social Partner Strategy  | ID          |
| :--------------- |:---------------:|
| Frontier  |   0        |
| Frontier Terminal  | 1             |
| Frontier and Beyond  | 2          |
| Beyond  | 3          |

The following line trains the GANGSTR agent with social intervention ratio of 20% and a Frontier and Beyond Strategy

```mpirun -np 24 python train.py --env-name FetchManipulate5Objects-v0 --intervention-prob 0.2 --strategy 2```

To change the ratio of social intervention, you can change the value of the _intervention-prob_ argument. Values should be within the interval [0, 1].

When _intervention-prob_ is equal to 0, the GANGSTR agent only performs individual learning. By contrast, when _intervention-prob_ is equal to 1, only social episodes are conducted. 
