package de.tu_darmstadt.ke.seco;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithmFactory;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.FMeasure;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Precision;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Recall;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRuleSet;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.Weka379AdapterMultilabel;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;
import weka.core.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.SystemMenuBar;

import com.sun.xml.internal.bind.v2.runtime.RuntimeUtil.ToStringAdapter;

/**
 * Runner class for building multilabel learners using XML config files, running it on a given dataset and outputting
 * the learned theory.
 *
 * @author Michael Rapp
 */
public class Main {

	public static String name;
	public static String path;
	public static List<Measure> eval_results;
	
    private static void evaluate(final MultiLabelInstances trainingData, final MultiLabelInstances testData,
                                 final MultiLabelLearner multilabelLearner) throws Exception {
    	System.out.println("Learned Model:\n");
        System.out.println(multilabelLearner);

        if (testData != null) {
            Evaluator evaluator = new Evaluator();
            Evaluation evaluation = evaluator.evaluate(multilabelLearner, trainingData, trainingData);
            System.out.println("\n\nEvaluation Results on train data:\n");
            System.out.println(evaluation);
            
            eval_results = evaluation.getMeasures();
            
            evaluator = new Evaluator();
            evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
            System.out.println("\n\nEvaluation Results on test data:\n");
            System.out.println(evaluation);
            
            eval_results.addAll(evaluation.getMeasures());
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
     *             -betaValue [optional]:
     *             		Beta value for F-Measure, where 0 <= Beta < 1 means more Precision and 1 < Beta means more Recall (default: -1 for no usage of F-Measure)
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
     *                 Whether the rules should be learned with bottom-up or not (default: false)
     *             -acceptEqual [optional]:
     *                 Whether a generalized rule should be accepted when it's equally good or only if it's better (default: true)
     *             -useSeCo [optional]:
     *             	   Whether the Separate And Conquer approach should be used or not (default: true)
     *             -n_step [optional]:
     *             		Whether the generalization should be done n times, no matter if the generated rules are better or not. (default: 0)
     *             -useRandom [optional]:
     *             		Whether the generalization should chose a random condition to generalize.
     *             -beamWidth [optional]:
     *             		The beamWidth that should be used (default: 1)
     *             -numericGeneralization [optional]:
     *             		How a numeric condition should be generalized (default: random; such that a random value between the current and the upper/lower bound is chosen)
     *             -coverAllLabels [optional]:
     *             		Whether a covered instance should be readded if not all of its labels are covered. (default: false)
     *             -evaluationMethod [optional]:
     *             		How the rules of the learned model should be combined to classify the test data
     * @throws Exception The exception, which is thrown, if any error occurs
     */
    public static void main(final String[] args) throws Exception {
        final String baseLearnerConfigPath = getMandatoryArgument("baselearner", args);
        final double betaValue = Double.valueOf(getOptionalArgument("beta", args, "-1"));
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
                EvaluationStrategy.RULE_INDEPENDENT);
        final String averagingStrategy = getOptionalArgument("averagingStrategy", args,
                AveragingStrategy.MICRO_AVERAGING);
        final boolean useBottomUp = Boolean.valueOf(getOptionalArgument("useBottomUp", args, "false"));
        final boolean acceptEqual = Boolean.valueOf(getOptionalArgument("acceptEqual", args, "true"));
        final boolean useSeCo = Boolean.valueOf(getOptionalArgument("useSeCo", args, "true"));
        final int n_step = Integer.valueOf(getOptionalArgument("n_step", args, "0"));
        final boolean useRandom = Boolean.valueOf(getOptionalArgument("useRandom", args, "false"));
        final String beamWidth = getOptionalArgument("beamWidth", args, "1");
        final String numericGeneralization = getOptionalArgument("numeric", args, "not-random");
        final boolean coverAllLabels = Boolean.valueOf(getOptionalArgument("coverAllLabels", args, "false"));
        final String evaluationMethod = getOptionalArgument("evaluationMethod", args, "DecisionList");

        
        // create file name from parameters for result
        
        Date date = new Date();
        String dateString = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss'.txt'").format(new Date());
        
        path = ""; //(useBottomUp ? "BottomUp\\" : "TopDown\\") + evaluationMethod + "\\";
        name = xmlLabelsDefFilePath.substring(5, xmlLabelsDefFilePath.length() - 4) + "_" + betaValue + "_" + dateString;
        
        //out = new PrintStream(new File("C:\\Users\\Pascal\\Documents\\Studium\\BachelorOfScienceInformatik\\Bachelorarbeit\\Experimente\\BottomUp\\Multiclass SeCo Multiclass DecisionList RuleIndependent\\Baseline Standard No NStep Not Random\\" + path + name));
        //PrintStream out = new PrintStream(new File("C:\\Users\\Pascal\\Documents\\Studium\\BachelorOfScienceInformatik\\Bachelorarbeit\\Ergebnisse\\" + name));
        
        
        System.setOut(System.out);
        
        //System.setOut(System.out);
        
        
        System.out.println("Arguments:\n");
        System.out.println("-baselearner " + baseLearnerConfigPath);
        System.out.println("-beta " + betaValue);
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
        System.out.println("-beamWidth " + beamWidth);
        System.out.println("-numericGeneralization " + numericGeneralization);
        System.out.println("-coverAllLabels " + coverAllLabels);
        System.out.println("-evaluationMethod " + evaluationMethod);
        System.out.println("\n");
        
        // Create training instances from dataset
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);

        
        
        
        
        System.out.println("SeCo: start experiment\n");

        
        SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
        Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm, betaValue, 
                remainingInstancesPercentage, reAddAllCovered, skipThresholdPercentage, predictZeroRules,
                useMultilabelHeads, evaluationStrategy, averagingStrategy, useBottomUp, acceptEqual, useSeCo, 
                n_step, useRandom, beamWidth, numericGeneralization, coverAllLabels, evaluationMethod);
        
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
        
        Results result = new Results();
        result.printResults(remainingInstancesPercentage, multilabelLearner, eval_results);

    }


}