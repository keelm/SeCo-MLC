# README

This repository provides the code of a separate-and-conquer rule-learning algorithm for learning multi-label head rules. The following instructions provide an overview on how to use the project.

The project provides the main class `de.tu_darmstadt.ke.seco.Main` for running the rule learner. Alternatively, the pre-built JAR-file `SeCo-MLC.jar` can be executed. The rule learner requires to specify several command line arguments, which are listed in the following:

| Argument                     | Description                                                                                                                       |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| baselearner                  | Path to the XML config file, which specifies the metric to use for selecting candidate rules (e.g. `/config/precision.xml`)       |
| arff                         | Path to the training data set in Weka .arff format (e.g. `/data/genbase-train.arff`)                                              |
| xml                          | Path to XML file containing labels meta-data (e.g. `/data/genbase.xml`)                                                           |
| test-arff                    | Path to the test data set in Weka .arff format (e.g. `/data/genbase-test.arff`)                                                   |
| remainingInstancesPercentage | The percentage of the training data set, which must not be covered for the algorithm to terminate (e.g. `0.1`)                    |
| readdAllCovered               | Whether fully-covered rules should be provided to the next separate-and-conquer iteration or not (must be `true` or `false`)      |
| skipThresholdPercentage      | The threshold, which should be used to create stopping rules. When set to a value < 0 no stopping rules are used (e.g. `0.01`)    |
| predictZeroRules             | Whether zero rules (predicting absent labels) should be learned or not (must be `true` or `false`)                                |
| useMultilabelHeads           | Whether multi-label head rules should be learned or not (must be `true` or `false`)                                               |
| averagingStrategy            | The averaging strategy to use (must be `micro-averaging`, `label-based-averaging`, `example-based-averaging` or `macro-averaging` |
| evaluationStrategy           | The evaluation strategy to use (must be `rule-dependent` or `rule-independent`)												   |
| useRelaxedPruning            | Whether or not to use relaxed pruning for discovering multilabel rules                                                            |
| useBoostedHeuristicForRules  | Whether or not to use the boosted heuristic value for evaluating rules if using relaxed pruning                                   |
| boostFunction                | The boost function to use for relaxed pruning (must be ´peak´, ´root´ or ´lln´)												   |
| label                        | At which number of labels the boost is fixed, is the peak/maximum for the peak boost function (e.g. 3.0)                          |
| boostAtLabel                 | The boost at the specified label (e.g. 1.1)   																					   |
| curvature                    | Curvature of the peak boost function                                                                                              |
| pruningDepth                 | Pruning depth if only using a anti-monotonic evaluation metric, -1 if guarantee best head (else e.g. 3)                           |                                                                                               |

In the following an exemplary command line argument for running the provided JAR file is given:

```
java -jar SeCo-MLC.jar -baselearner config/precision.xml -arff data/genbase-train.arff -xml data/genbase.xml -test-arff data/genbase-test.arff -remainingInstancesPercentage 0.1 -readdAllCovered true -skipThresholdPercentage 0.01 -predictZeroRules true -useMultilabelHeads true -averagingStrategy micro-averaging -evaluationStrategy rule-dependent  -useRelaxedPruning true -useBoostedHeuristicForRules true -boostFunction peak -label 3.0 -boostAtLabel 1.1 -curvature 2.0 -pruningDepth -1
```