== Design ==

```
NpzGeneratorDataset(datasets/npy_generator.py)
_load, load
              ^
              |
H5fGeneratorDataset
_load



AbstractAgentBasedModel (abstract.py)
trainGenerator, testGenerator, _yieldLoop, convert, save, load, _loadWeights
predict, getOptimizer
               ^
               |
HierarchicalAgentBasedModel (abstract.py)
_makeHierarchicalModel, save, _loadWeights, predict

               ^
               |
RobotMultiHierarchical (multi_hierarchical.py)
_makeModel, _makeSimpleActor, _makeConditionalActor, _makeAll, _getData,
_loadWeights, save, trainFromGenerators, _sizes
               ^
               |
RobotMultiPredictionSampler (multi_sampler.py)
_makePredictor, _makeTransform, _getTransform, _makeDenseTransform, _makeModel
_getData, trainFromGenerators, save, predict, _loadWeights, _makeGenerator,
_makeEncoder, _makeDecoder, _makeActorPolicy, _makeStateEncoder
_targetsFromTrainTargets, validate, encode, decode, transform
               ^
               |
PretrainImageAutoEncoder (pretrain_image.py)
_makePredictor, _getData
               ^
               |
PretrainImageCostar (pretrain_image_costar.py)
_makePredictor, _makeModel, _getData

multi.py:
MakeImageClassifier, MakeImageEncoder, GetPoseModel, GetActorModel
MakeMultiPolicy, MakeImageDecoder, GetAllMultiData

costar.py:
MakeCostarImageClassifier


ConditionalImage
  ^
  |
ConditionalImageCostar (conditional_image_costar.py)
_makeModel, _getData
```

== Flow ==
- scripts/ctp_model_tool: ingress point
  - parse.py: arguments
  - dataset = H5fGeneratorDataset (h5f_generator.py)
    - most functionality is in npy_generator.py
  - data = dataset.load; data is a sample
  - gen = model.trainGenerator(dataset) (abstract.py)
    - _yieldLoop(dataset.sampleTrain)
      - _getData()
  - trainFromGenerators(gen)
    - Feeds data (features) to PredictorCb
      - The ImageCb will always use the same bit of data

== Design Issues ==
- There are multiple ways of reading from a data file: by keys, by a whole column, keeping the hp5 format, converting to ndarray.
  - dataset.load() returns a single file sample in a dictionary, but needs the processing logic for it.
  - The real logic sits in the AbstractAgentBasedModel class in abstract.py
  - The sample is used to create the models, so duplication results
