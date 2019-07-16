package de.tu_darmstadt.ke.seco;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithmFactory;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.Weka379AdapterMultilabel;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.core.Utils;

import java.io.File;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.SystemMenuBar;

/**
 * Runner class for building multilabel learners using XML config files, running it on a given dataset and outputting
 * the learned theory.
 *
 * @author Michael Rapp
 */
public class Main {

	public static String name;
	
    private static void evaluate(final MultiLabelInstances trainingData, final MultiLabelInstances testData,
                                 final MultiLabelLearner multilabelLearner) throws Exception {
    	System.out.println("Learned Model:\n");
        System.out.println(multilabelLearner);

        if (testData != null) {
            Evaluator evaluator = new Evaluator();
            Evaluation evaluation = evaluator.evaluate(multilabelLearner, trainingData, trainingData);
            System.out.println("\n\nEvaluation Results on train data:\n");
            System.out.println(evaluation);
            evaluator = new Evaluator();
            evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
            System.out.println("\n\nEvaluation Results on test data:\n");
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
     *             -remainingInstancesPercentage [optional]:
     *                 The percentage of the training dataset which must not be covered when the algorithm terminates
     *             -reAddAllCovered [optional]:
     *                 Whether all already covered rules should be used for the next separate-and-conquer iteration or
     *                 not
     *             -skipThresholdPercentage [optional]:
     *                 The threshold, which should be used for stopping rules. When set to a value < 0 no stopping rules
     *                 are used
     *             -predictZeroRules [optional]:
     *                 Whether zero rules should be predicted or not
     *             -useMultilabelHeads [optional]:
     *                 Whether multi-label head rule should be learned or not
     *             -useBottomUp [optional]:
     *                 Whether the rules should be learned with bottom-up or not
     *             -acceptEqual [optional]:
     *                 Whether a generalized rule should be accepted when it's equally good or only if it's better (default: true)
     *             -useSeCo [optional]:
     *             	   Whether the Separate And Conquer approach should be used or not (default: true)
     * @throws Exception The exception, which is thrown, if any error occurs
     */
    public static void main(final String[] args) throws Exception {
        final String baseLearnerConfigPath = getMandatoryArgument("baselearner", args);
        final String arffFilePath = getMandatoryArgument("arff", args);
        final String xmlLabelsDefFilePath = getOptionalArgument("xml", args, arffFilePath.replace(".arff", ".xml"));
        final String testArffFilePath = getOptionalArgument("test-arff", args, null);
        final double remainingInstancesPercentage = Double
                .valueOf(getOptionalArgument("remainingInstancesPercentage", args, "0.05"));
        final boolean reAddAllCovered = Boolean.valueOf(getOptionalArgument("reAddAllCovered", args, "false"));
        final double skipThresholdPercentage = Double
                .valueOf(getOptionalArgument("skipThresholdPercentage", args, "-1.0"));
        final boolean predictZeroRules = Boolean.valueOf(getOptionalArgument("predictZeroRules", args, "false"));
        final boolean useMultilabelHeads = Boolean.valueOf(getOptionalArgument("useMultilabelHeads", args, "false"));
        final String evaluationStrategy = getOptionalArgument("evaluationStrategy", args,
                EvaluationStrategy.RULE_DEPENDENT);
        final String averagingStrategy = getOptionalArgument("averagingStrategy", args,
                AveragingStrategy.MICRO_AVERAGING);
        final boolean useBottomUp = Boolean.valueOf(getOptionalArgument("useBottomUp", args, "false"));
        final boolean acceptEqual = Boolean.valueOf(getOptionalArgument("acceptEqual", args, "true"));
        final boolean useSeCo = Boolean.valueOf(getOptionalArgument("useSeCo", args, "true"));
        final int n_step = Integer.valueOf(getOptionalArgument("n_step", args, "1"));
        final boolean useRandom = Boolean.valueOf(getOptionalArgument("useRandom", args, "false"));
        final String evaluationMethod = getOptionalArgument("evaluationMethod", args, "DNF");

        
        // create file name from parameters for result
        Date date = new Date();
        String dateString = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss'.txt'").format(new Date());
        name = "BU_" + useBottomUp + "Eq_" + acceptEqual + "SeCo_" + useSeCo + "_" + dateString;
        PrintStream out = new PrintStream(new File("C:\\Users\\Pascal\\Documents\\Studium\\BachelorOfScienceInformatik\\Bachelorarbeit\\Experimente\\weather_precision\\" + name));
        System.setOut(out);
        
        System.out.println("Arguments:\n");
        System.out.println("-baselearner " + baseLearnerConfigPath);
        System.out.println("-arff " + arffFilePath);
        System.out.println("-xml " + xmlLabelsDefFilePath);
        System.out.println("-test-arff " + testArffFilePath);
        System.out.println("-remainingInstancesPercentage " + remainingInstancesPercentage);
        System.out.println("-reAddAllCovered " + reAddAllCovered);
        System.out.println("-skipThresholdPercentage " + skipThresholdPercentage);
        System.out.println("-predictZeroRules " + predictZeroRules);
        System.out.println("-useMultilabelHeads " + useMultilabelHeads);
        System.out.println("-evaluationStrategy " + evaluationStrategy);
        System.out.println("-averagingStrategy " + averagingStrategy);
        System.out.println("-useBottomUp " + useBottomUp);
        System.out.println("-acceptEqual " + acceptEqual);
        System.out.println("-useSeCo " + useSeCo);
        System.out.println("-n_step " + n_step);
        System.out.println("-useRandom " + useRandom);
        System.out.println("-evaluationMethod " + evaluationMethod);
        System.out.println("\n");
        
        // Create training instances from dataset
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);

        System.out.println("SeCo: start experiment\n");

        
        SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
        Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                remainingInstancesPercentage, reAddAllCovered, skipThresholdPercentage, predictZeroRules,
                useMultilabelHeads, evaluationStrategy, averagingStrategy, useBottomUp, acceptEqual, useSeCo, n_step, useRandom, evaluationMethod);

        // Create test instances from dataset, if available
        final MultiLabelInstances testData =
                testArffFilePath != null ? new MultiLabelInstances(testArffFilePath, xmlLabelsDefFilePath) : null;

        // Learn model from training instances
        long startTime = System.currentTimeMillis();
        multilabelLearner.build(trainingData);
        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("building the model took secs: "+estimatedTime/1000.0);

        // Evaluate model on test instances, if available
        startTime = System.currentTimeMillis();
        evaluate(trainingData, testData, multilabelLearner);
        estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("evaluating the model took secs: "+estimatedTime/1000.0);
        System.out.println("SeCo: finish experiment\n");

    }


}