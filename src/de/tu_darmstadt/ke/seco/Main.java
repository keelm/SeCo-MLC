package de.tu_darmstadt.ke.seco;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithmFactory;
import de.tu_darmstadt.ke.seco.models.MultiHeadRuleSet;
import de.tu_darmstadt.ke.seco.models.RuleSet;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.MultiLabelPostProcessor;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.Weka379AdapterMultilabel;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Random;

/**
 * Runner class for building multilabel learners using XML config files, running it on a given dataset and outputting
 * the learned theory.
 *
 * @author Michael Rapp
 */
public class Main {

    private static void evaluate(final MultiLabelInstances trainingData, final MultiLabelInstances testData,
                                 final MultiLabelLearner multilabelLearner) throws Exception {
        System.out.println("Learned Model:\n");
        System.out.println(multilabelLearner);

        if (testData != null) {
            Evaluator evaluator = new Evaluator();
            Evaluation evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
            System.out.println("\n\nEvaluation Results:\n");
            System.out.println(evaluation);
        }
    }

    private static String getMandatoryArgument(final String argument, final String[] args) throws Exception {
        String value = Utils.getOption(argument, args);

        if (value.isEmpty()) {
            throw new IllegalArgumentException("Mandatory argument -" + argument + " missing");
        }

        return value;
    }

    private static String getOptionalArgument(final String argument, final String[] args,
                                              final String defaultValue) throws Exception {
        String value = Utils.getOption(argument, args);

        if (value.isEmpty()) {
            return defaultValue;
        }

        return value;
    }

    /**
     * Builds a multilabel learner according to a specific XML config files, runs it of the specified dataset and
     * outputs the learned theory.
     *
     * @param args -baselearner: Path to base learner XML config file (e.g. /config/new_config/ripper/Ripper.xml) -arff:
     *             Path to training dataset in the weka .arff format (e.g. /data/multilabel/scene.arff) -xml [optional]:
     *             Path to XML file containing labels meta-data (e.g. /data/multilabel/scene.xml)
     *             -remainingInstancesPercentage [optional]: The percentage of the training dataset which must not be
     *             covered when the algorithm terminates -readAllCovered [optional]: Whether all already covered rules
     *             should be used for the next separate-and-conquer iteration or not -skipThresholdPercentage
     *             [optional]: The threshold, which should be used for stopping rules. When set to a value < 0 no
     *             stopping rules are used -predictZeroRules [optional]: Whether zero rules should be predicted or not
     *             -useMultilabelHeads [optional]: Whether multi-label head rule should be learned or not -beamWidth
     *             [optional]: The beam width to use when learning multi-label head rules. Must be an integer between 1
     *             and the number of attributes, or 0.0 and 1.0 -postProcessing [optional]: Whether the learned model
     *             should be post-processed or not -bias: The bias, which should be used to weight multi-label head
     *             rules depending on their head's size. Must be >= 1
     * @throws Exception The exception, which is thrown, if any error occurs
     */
    public static void main(final String[] args) throws Exception {
        final String baseLearnerConfigPath = getMandatoryArgument("baselearner", args);
        final String arffFilePath = getMandatoryArgument("arff", args);
        final String xmlLabelsDefFilePath = getOptionalArgument("xml", args, arffFilePath.replace(".arff", ".xml"));
        final String testArffFilePath = getOptionalArgument("test-arff", args, null);
        final double remainingInstancesPercentage = Double
                .valueOf(getOptionalArgument("remainingInstancesPercentage", args, "0.05"));
        final boolean readAllCovered = Boolean.valueOf(getOptionalArgument("readAllCovered", args, "false"));
        final double skipThresholdPercentage = Double
                .valueOf(getOptionalArgument("skipThresholdPercentage", args, "-1.0"));
        final boolean predictZeroRules = Boolean.valueOf(getOptionalArgument("predictZeroRules", args, "false"));
        final boolean useMultilabelHeads = Boolean.valueOf(getOptionalArgument("useMultilabelHeads", args, "false"));
        final String beamWidth = getOptionalArgument("beamWidth", args, "1");
        final String evaluationStrategy = getOptionalArgument("evaluationStrategy", args,
                EvaluationStrategy.RULE_DEPENDENT);
        final String averagingStrategy = getOptionalArgument("averagingStrategy", args,
                AveragingStrategy.MICRO_AVERAGING);
        final boolean postProcessing = Boolean.valueOf(getOptionalArgument("postProcessing", args, "false"));
        final double bias = Double.valueOf(getOptionalArgument("bias", args, "1.0"));
        final String cv = getOptionalArgument("cv", args, null);
        final String iterations = getOptionalArgument("iterations", args, null);

        System.out.println("Arguments:\n");
        System.out.println("-baselearner " + baseLearnerConfigPath);
        System.out.println("-arff " + arffFilePath);
        System.out.println("-xml " + xmlLabelsDefFilePath);
        System.out.println("-test-arff " + testArffFilePath);
        System.out.println("-remainingInstancesPercentage " + remainingInstancesPercentage);
        System.out.println("-readAllCovered " + readAllCovered);
        System.out.println("-skipThresholdPercentage " + skipThresholdPercentage);
        System.out.println("-predictZeroRules " + predictZeroRules);
        System.out.println("-useMultilabelHeads " + useMultilabelHeads);
        System.out.println("-beamWidth " + beamWidth);
        System.out.println("-evaluationStrategy " + evaluationStrategy);
        System.out.println("-averagingStrategy " + averagingStrategy);
        System.out.println("-postProcessing " + postProcessing);
        System.out.println("-bias " + bias);
        System.out.println("-cv " + cv);
        System.out.println("\n");

        // Create training instances from dataset
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);

        SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
        Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                remainingInstancesPercentage, readAllCovered, skipThresholdPercentage, predictZeroRules,
                useMultilabelHeads, beamWidth, evaluationStrategy, averagingStrategy, bias);

        if (cv == null || iterations == null) {
            // Create test instances from dataset, if available
            final MultiLabelInstances testData =
                    testArffFilePath != null ? new MultiLabelInstances(testArffFilePath, xmlLabelsDefFilePath) : null;

            // Learn model from training instances
            multilabelLearner.build(trainingData);

            // Evaluate model on test instances, if available
            evaluate(trainingData, testData, multilabelLearner);

            // Post process model and evaluate again
            if (postProcessing) {
                RuleSet<?> model = multilabelLearner.getSeCoClassifier().getTheory();

                if (model instanceof MultiHeadRuleSet) {
                    System.out.println("Post-processing model...\n");
                    MultiLabelPostProcessor postProcessor = new MultiLabelPostProcessor();
                    MultiHeadRuleSet postProcessedModel = postProcessor.postProcess((MultiHeadRuleSet) model);
                    multilabelLearner.getSeCoClassifier().setTheory(postProcessedModel);
                    evaluate(trainingData, testData, multilabelLearner);
                }
            }
        } else {
            makeExhaustiveCrossValidation(multilabelLearner, trainingData, Integer.valueOf(cv),
                    Integer.valueOf(iterations));
        }
    }

    private static void makeExhaustiveCrossValidation(MultiLabelLearner multiLabelLearnerBase,
                                                      MultiLabelInstances dataSet, final int fold,
                                                      final int numFolds) throws
            Exception {
        int seed = 1;

        Instances workingSet = new Instances(dataSet.getDataSet());
        Random random = new Random(seed);
        workingSet.randomize(random);

        MultiLabelLearner multiLabelLearner;

        try {
            // if test on train
            Instances train;
            Instances test;
            MultiLabelInstances mlTrain = dataSet;
            MultiLabelInstances mlTest = dataSet;
            //if fold of CV
            if (fold != 0) {
                train = workingSet.trainCV(numFolds, fold - 1);
                test = workingSet.testCV(numFolds, fold - 1);
                mlTrain = new MultiLabelInstances(train, dataSet.getLabelsMetaData());
                mlTest = new MultiLabelInstances(test, dataSet.getLabelsMetaData());
            }

            multiLabelLearner = multiLabelLearnerBase.makeCopy();

            if (fold == 0)
                System.out.println("Train on whole training set ");
            else
                System.out.println("Train fold " + (fold));

            long before = System.currentTimeMillis();
            multiLabelLearner.build(mlTrain);
            long after = System.currentTimeMillis();
            double buildTime = (after - before) / 1000000.0;  //changed to milliseconds

            System.out.println("==============================");
            if (fold == 0)
                System.out.println(
                        "Model and Results on whole training set (model after first evaluation / first iteration)");
            else
                System.out.println(
                        "Model and Results on fold " + fold + " (model after first evaluation / first iteration)");
            System.out.println("==============================");
            System.out.println("buildTime: " + buildTime);
            System.out.println("Learned Model:\n");
            System.out.println(multiLabelLearner);
            Evaluator evaluator = new Evaluator();
            Evaluation evaluation = evaluator.evaluate(multiLabelLearner, mlTest, mlTrain);
            System.out.println("\n\nEvaluation Results:\n");
            System.out.println(evaluation);

        } catch (Exception e) {
            System.err.println("Error on CV fold " + fold);
            e.printStackTrace();
        }
    }


}