# Separate-and-Conquer Multi-label Rule Learning

This repository provides the code of a separate-and-conquer rule-learning algorithm for learning multi-label head rules. The following instructions provide an overview on how to use the project.

If you want to use the code, or just cite the paper, please use the following:
```
@INPROCEEDINGS{mr:ML-Antimonotonicity,
     author = {Rapp, Michael and Loza Menc{\'{\i}}a, Eneldo and F{\"{u}}rnkranz, Johannes},
     editor = {Phung, Dinh Q. and Tseng, Vincent S. and Webb, Geoffrey I. and Ho, Bao and Ganji, Mohadeseh and Rashidi, Lida},
      title = {Exploiting Anti-monotonicity of Multi-label Evaluation Measures for Inducing Multi-label Rules},
  booktitle = {PAKDD 2018: Advances in Knowledge Discovery and Data Mining},
       year = {2018},
      pages = {29--42},
  publisher = {Springer International Publishing},
    address = {Cham},
       isbn = {978-3-319-93034-3},
        url = {https://arxiv.org/abs/1812.06833},
        doi = {10.1007/978-3-319-93034-3_3},
}
@ARTICLE{loza16MLRL,
    author = {Loza Menc{\'{\i}}a, Eneldo and Janssen, Frederik},
    editor = {D{\v z}eroski, Sa{\v s}o and Kocev, Dragi and Panov, Pan{\v c}e},
  keywords = {Label Dependencies, multilabel classification, Rule Learning, Stacking},
     month = may,
     title = {Learning rules for multi-label classification: a stacking and a separate-and-conquer approach},
   journal = {Machine Learning},
    volume = {105},
    number = {1},
      year = {2016},
     pages = {77--126},
      issn = {0885-6125},
       url = {https://www.ke.tu-darmstadt.de/publications/papers/loza16MLRL.pdf},
       doi = {10.1007/s10994-016-5552-1},
}
``` 

Check also out the branch relaxed-pruning (https://github.com/keelm/SeCo-MLC/tree/relaxed-pruning) for the technique described in
``` 
@INPROCEEDINGS{yk:Relaxed-Pruning,
     author = {Klein, Yannik and Rapp, Michael and Loza Menc{\'{\i}}a, Eneldo},
     editor = {Kralj Novak, Petra and {\v S}muc, Tomislav and D{\v z}eroski, Sa{\v s}o},
   keywords = {Label Dependencies, multilabel classification, Rule Learning},
      month = oct,
      title = {Efficient Discovery of Expressive Multi-label Rules using Relaxed Pruning},
  booktitle = {Discovery Science},
       year = {2019},
      pages = {367--382},
  publisher = {Springer International Publishing},
       note = {Best Student Paper Award},
       isbn = {978-3-030-33778-0},
        url = {https://arxiv.org/abs/1908.06874},
        doi = {10.1007/978-3-030-33778-0_28},
}
``` 


The project provides the main class `de.tu_darmstadt.ke.seco.Main` for running the rule learner. Alternatively, the pre-built JAR-file `SeCo-MLC.jar` can be executed. The rule learner requires to specify several command line arguments, which are listed in the following:

| Argument                     | Description                                                                                                                       |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| baselearner                  | Path to the XML config file, which specifies the metric to use for selecting candidate rules (e.g. `/config/precision.xml`)       |
| arff                         | Path to the training data set in Weka .arff format (e.g. `/data/genbase-train.arff`)                                              |
| xml                          | Path to XML file containing labels meta-data (e.g. `/data/genbase.xml`)                                                           |
| test-arff                    | Path to the test data set in Weka .arff format (e.g. `/data/genbase-test.arff`)                                                   |
| remainingInstancesPercentage | The percentage of the training data set, which must not be covered for the algorithm to terminate (e.g. `0.1`)                    |
| readAllCovered               | Whether fully-covered rules should be provided to the next separate-and-conquer iteration or not (must be `true` or `false`)      |
| skipThresholdPercentage      | The threshold, which should be used to create stopping rules. When set to a value < 0 no stopping rules are used (e.g. `0.01`)    |
| predictZeroRules             | Whether zero rules (predicting absent labels) should be learned or not (must be `true` or `false`)                                |
| useMultilabelHeads           | Whether multi-label head rules should be learned or not (must be `true` or `false`)                                               |
| averagingStrategy            | The averaging strategy to use (must be `micro-averaging`, `label-based-averaging`, `example-based-averaging` or `macro-averaging` |
| evaluationStrategy           | The evaluation strategy to use (must be `rule-dependent` or `rule-independent`                                                    |

In the following an exemplary command line argument for running the provided JAR file is given:

```
java -jar SeCo-MLC.jar -baselearner config/precision.xml -arff data/genbase-train.arff -xml data/genbase.xml -arff-test data/genbase-test.arff -remainingInstancesPercentage 0.1 -readAllCovered true -skipThresholdPercentage 0.01 -predictZeroRules true -useMultilabelHeads true -averagingStrategy micro-averaging -evaluationStrategy rule-dependent
```