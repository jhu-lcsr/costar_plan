# Other Domains

These are partially-supported domains that may be of interest.

### Grid World

[![Grid World](https://img.youtube.com/vi/LLs1OIIIQnw/0.jpg)](https://youtu.be/LLs1OIIIQnw)

Grid world is an ultra-simple driving task. The actor must move around a grid and navigate an intersection. Note that as the first domain implemented, this does not fully support the TTS API.

#### Run

```
create_training_data.py
learn_simple.py
```

### Needle Master

[![Needle Master Gameplay](https://img.youtube.com/vi/GgIznhbk-5g/0.jpg)](https://youtu.be/GgIznhbk-5g)

Example from a simple Android game. This comes with a dataset that can be used; the goal is to generate task and motion plans that align with the expert training data.

One sub-task from the Needle Master domain is trajectory optimization. The goal is to generate an optimal trajectory in the shortest possible amount of time.

References:
```
@inproceedings{paxton2016towards,
  title={Towards Robot Task Planning from Probabilistic Representations of Human Skills},
  author={Paxton, Chris and Kobilarov, Marin and Hager, Gregory D.},
  booktitle={AAAI 2016 Workshop on Planning in Hybrid Systems},
  year={2016},
  organization={AAAI}
}

@inproceedings{paxton2016want,
  title={Do what I want, not what I did: Imitation of skills by planning sequences of actions},
  author={Paxton, Chris and Jonathan, Felix and Kobilarov, Marin and Hager, Gregory D},
  booktitle={Intelligent Robots and Systems (IROS), 2016 IEEE/RSJ International Conference on},
  pages={3778--3785},
  year={2016},
  organization={IEEE}
}
```
