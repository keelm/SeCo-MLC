package de.tu_darmstadt.ke.seco;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithmFactory;
import de.tu_darmstadt.ke.seco.utils.Weka379Adapter;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.core.Utils;

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
     * @param args -baselearner:
     *                 Path to base learner XML config file (e.g. /config/precision.xml)
     *             -arff:
     *                 Path to training dataset in the weka .arff format (e.g. /data/genbase.arff)
     *             -xml [optional]:
     *                 Path to XML file containing labels meta-data (e.g. /data/genbase.xml)
     * @throws Exception The exception, which is thrown, if any error occurs
     */
    public static void main(final String[] args) throws Exception {
        final String baseLearnerConfigPath = getMandatoryArgument("baselearner", args);
        final String arffFilePath = getMandatoryArgument("arff", args);
        final String xmlLabelsDefFilePath = getOptionalArgument("xml", args, arffFilePath.replace(".arff", ".xml"));
        final String testArffFilePath = getOptionalArgument("test-arff", args, null);

        System.out.println("Arguments:\n");
        System.out.println("-baselearner " + baseLearnerConfigPath);
        System.out.println("-arff " + arffFilePath);
        System.out.println("-xml " + xmlLabelsDefFilePath);
        System.out.println("-test-arff " + testArffFilePath);
        System.out.println("\n");

        // Create training instances from dataset
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);

        SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
        Weka379Adapter weka379Adapter = new Weka379Adapter(baseLearnerAlgorithm);
        BinaryRelevance binaryRelevance = new BinaryRelevance(weka379Adapter);

        // Create test instances from dataset, if available
        final MultiLabelInstances testData =
                testArffFilePath != null ? new MultiLabelInstances(testArffFilePath, xmlLabelsDefFilePath) : null;

        // Learn model from training instances
        binaryRelevance.build(trainingData);

        // Evaluate model on test instances, if available
        evaluate(trainingData, testData, binaryRelevance);
    }


}