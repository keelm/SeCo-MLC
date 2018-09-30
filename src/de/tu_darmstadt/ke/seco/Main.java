package de.tu_darmstadt.ke.seco;

import com.opencsv.CSVWriter;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithmFactory;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.MainEvaluation;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.Weka379AdapterMultilabel;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.Weka379AdapterPrepending;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;
import weka.core.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;

import static de.tu_darmstadt.ke.seco.multilabelrulelearning.MulticlassCovering.*;

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
            Evaluation trainingEvaluation = evaluator.evaluate(multilabelLearner, trainingData, trainingData);
            System.out.println("\n\nEvaluation Results on train data:\n");
            System.out.println(trainingEvaluation);
            evaluator = new Evaluator();
            Evaluation evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
            for (Measure measure : evaluation.getMeasures()) {
                csvWriter.writeNext(new String[]{measure.getName(), Double.toString(measure.getValue())});
            }
            System.out.println("\n\nEvaluation Results on test data:\n");
            System.out.println(evaluation);


            csvWriter.writeNext(new String[]{"avg. #evals per findBestHead()", Double.toString(evaluationsPerHead)});
            csvWriter.writeNext(new String[]{"#findBestHead()", Integer.toString(evaluatedHeads)});

            for (Measure measure : trainingEvaluation.getMeasures()) {
                csvWriter.writeNext(new String[]{measure.getName(), Double.toString(measure.getValue())});
            }
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
     *             -readAllCovered [optional]:
     *                 Whether all already covered rules should be used for the next separate-and-conquer iteration or
     *                 not
     *             -skipThresholdPercentage [optional]:
     *                 The threshold, which should be used for stopping rules. When set to a value < 0 no stopping rules
     *                 are used
     *             -predictZeroRules [optional]:
     *                 Whether zero rules should be predicted or not
     *             -useMultilabelHeads [optional]:
     *                 Whether multi-label head rule should be learned or not
     * @throws Exception The exception, which is thrown, if any error occurs
     */
    public static void main(final String[] args) throws Exception {
        final boolean eval = Boolean.valueOf(getOptionalArgument("evaluate", args, "false"));
        if (eval) {
            MainEvaluation.createTasks(args);
            return;
        }

        final String baseLearnerConfigPath = getMandatoryArgument("baselearner", args);
        final String arffFilePath = getMandatoryArgument("arff", args);
        final String xmlLabelsDefFilePath = getOptionalArgument("xml", args, arffFilePath.replace(".arff", ".xml"));
        final String testArffFilePath = getOptionalArgument("test-arff", args, null);
        final double remainingInstancesPercentage = Double.valueOf(getOptionalArgument("remainingInstancesPercentage", args, "0.05"));
        final boolean readdAllCovered = Boolean.valueOf(getOptionalArgument("readdAllCovered", args, "false"));
        final double skipThresholdPercentage = Double.valueOf(getOptionalArgument("skipThresholdPercentage", args, "-1.0"));
        final boolean predictZeroRules = Boolean.valueOf(getOptionalArgument("predictZeroRules", args, "false"));
        final boolean useMultilabelHeads = Boolean.valueOf(getOptionalArgument("useMultilabelHeads", args, "false"));
        final String evaluationStrategy = getOptionalArgument("evaluationStrategy", args, EvaluationStrategy.RULE_DEPENDENT);
        final String averagingStrategy = getOptionalArgument("averagingStrategy", args, AveragingStrategy.MICRO_AVERAGING);
        // relaxed pruning options
        final boolean useRelaxedPruning = Boolean.valueOf((getOptionalArgument("useRelaxedPruning", args, "false")));
        final boolean useBoostedHeuristicForRules = Boolean.valueOf(getOptionalArgument("useBoostedHeuristicForRules", args, "true"));
        final String boostFunction = getOptionalArgument("boostFunction", args, "llm");
        final double label = Double.valueOf(getOptionalArgument("label", args, "3.0"));
        final double boostAtLabel = Double.valueOf(getOptionalArgument("boostAtLabel", args, "1.1"));
        final double curvature = Double.valueOf(getOptionalArgument("curvature", args, "2.0"));
        final int pruningDepth = Integer.valueOf(getOptionalArgument("pruningDepth", args, "-1"));
        final boolean usePrepending = Boolean.valueOf((getOptionalArgument("prepending", args, "false")));

        System.out.println("Arguments:\n");
        System.out.println("-baselearner " + baseLearnerConfigPath);
        System.out.println("-arff " + arffFilePath);
        System.out.println("-xml " + xmlLabelsDefFilePath);
        System.out.println("-test-arff " + testArffFilePath);
        System.out.println("-remainingInstancesPercentage " + remainingInstancesPercentage);
        System.out.println("-readdAllCovered " + readdAllCovered);
        System.out.println("-skipThresholdPercentage " + skipThresholdPercentage);
        System.out.println("-predictZeroRules " + predictZeroRules);
        System.out.println("-useMultilabelHeads " + useMultilabelHeads);
        System.out.println("-evaluationStrategy " + evaluationStrategy);
        System.out.println("-averagingStrategy " + averagingStrategy);
        System.out.println("-useRelaxedPruning " + useRelaxedPruning);
        System.out.println("-useBoostedHeuristicForRules " + useBoostedHeuristicForRules);
        System.out.println("-boostFunction " + boostFunction);
        System.out.println("-label " + label);
        System.out.println("-boostAtLabel " + boostAtLabel);
        System.out.println("-curvature " + curvature);
        System.out.println("-pruningDepth " + pruningDepth);
        System.out.println("-prepending " + usePrepending);
        System.out.println("\n");

        // create csv file
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss.SSS");
        filename = "results/experiments/" + xmlLabelsDefFilePath.split("/")[1].split("\\.")[0] + "_" + sdf.format(new Date()) + ".csv";
        File file = new File(filename);
        System.out.println(filename);
        file.createNewFile();
        fileWriter = new FileWriter(file);

        // create csv writer
        csvWriter = new CSVWriter(fileWriter,
                CSVWriter.DEFAULT_SEPARATOR,
                CSVWriter.NO_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END);

        // write settings to file
        String[] headerRecord = {"Info", "Value"};
        csvWriter.writeNext(headerRecord);
        csvWriter.writeNext(new String[]{"baselearner", baseLearnerConfigPath});
        csvWriter.writeNext(new String[]{"arff", arffFilePath});
        csvWriter.writeNext(new String[]{"xml", xmlLabelsDefFilePath});
        csvWriter.writeNext(new String[]{"test-arff", testArffFilePath});
        csvWriter.writeNext(new String[]{"remainingInstancesPercentage", Double.toString(remainingInstancesPercentage)});
        csvWriter.writeNext(new String[]{"readdAllCovered", Boolean.toString(readdAllCovered)});
        csvWriter.writeNext(new String[]{"skipThresholdPercentage", Double.toString(skipThresholdPercentage)});
        csvWriter.writeNext(new String[]{"predictZeroRules", Boolean.toString(predictZeroRules)});
        csvWriter.writeNext(new String[]{"useMultilabelHeads", Boolean.toString(useMultilabelHeads)});
        csvWriter.writeNext(new String[]{"evaluationStrategy", evaluationStrategy});
        csvWriter.writeNext(new String[]{"averagingStrategy", averagingStrategy});
        csvWriter.writeNext(new String[]{"useRelaxedPruning ", Boolean.toString(useRelaxedPruning)});
        csvWriter.writeNext(new String[]{"useBoostedHeuristicForRules", Boolean.toString(useBoostedHeuristicForRules)});
        csvWriter.writeNext(new String[]{"boostFunction", boostFunction});
        csvWriter.writeNext(new String[]{"label", Double.toString(label)});
        csvWriter.writeNext(new String[]{"boostAtLabel", Double.toString(boostAtLabel)});
        csvWriter.writeNext(new String[]{"curvature", Double.toString(curvature)});
        csvWriter.writeNext(new String[]{"pruningDepth", Integer.toString(pruningDepth)});

        // create training instances from dataset
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);

        System.out.println("SeCo: start experiment\n");

        MultiLabelLearnerBase multilabelLearner = null;
        if (usePrepending) {
            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            multilabelLearner = new Weka379AdapterPrepending(baseLearnerAlgorithm,
                    remainingInstancesPercentage, readdAllCovered, skipThresholdPercentage, predictZeroRules,
                    useMultilabelHeads, evaluationStrategy, averagingStrategy,
                    useRelaxedPruning, useBoostedHeuristicForRules, boostFunction, label, boostAtLabel, curvature, pruningDepth);
        } else {
            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                remainingInstancesPercentage, readdAllCovered, skipThresholdPercentage, predictZeroRules,
                useMultilabelHeads, evaluationStrategy, averagingStrategy,
                useRelaxedPruning, useBoostedHeuristicForRules, boostFunction, label, boostAtLabel, curvature, pruningDepth);
        }

        // Create test instances from dataset, if available
        final MultiLabelInstances testData = testArffFilePath != null ? new MultiLabelInstances(testArffFilePath, xmlLabelsDefFilePath) : null;

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
        fileWriter.close();
    }

    public static CSVWriter csvWriter;
    public static FileWriter fileWriter;
    public static String filename;

}