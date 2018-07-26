package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import com.opencsv.CSVWriter;
import de.tu_darmstadt.ke.seco.Main;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithmFactory;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;
import weka.core.Utils;

import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainEvaluationCopy {

    private static class Settings {

        public Settings(double value, String evaluationMeasureValue, String averagingStrategyValue, boolean predictZeroRulesValue,
                        boolean readdAllCoveredValue, double remainingInstancesPercentage, double skipThresholdPercentage) {
            this.value = value;
            this.evaluationMeasureValue = evaluationMeasureValue;
            this.averagingStrategyValue = averagingStrategyValue;
            this.predictZeroRulesValue = predictZeroRulesValue;
            this.readdAllCoveredValue = readdAllCoveredValue;
            this.remainingInstancesPercentage = remainingInstancesPercentage;
            this.skipThresholdPercentage = skipThresholdPercentage;
            this.useRelaxedPruning = false;
        }

        public Settings(double value, String evaluationMeasureValue, String averagingStrategyValue, boolean predictZeroRulesValue,
                        boolean readdAllCoveredValue, double remainingInstancesPercentage, double skipThresholdPercentage,
                        String boostFunctionValue, int labelValue, double boost, double curvature) {
            this.value = value;
            this.evaluationMeasureValue = evaluationMeasureValue;
            this.averagingStrategyValue = averagingStrategyValue;
            this.predictZeroRulesValue = predictZeroRulesValue;
            this.readdAllCoveredValue = readdAllCoveredValue;
            this.remainingInstancesPercentage = remainingInstancesPercentage;
            this.skipThresholdPercentage = skipThresholdPercentage;
            this.useRelaxedPruning = true;
            this.boostFunctionValue = boostFunctionValue;
            this.labelValue = labelValue;
            this.boost = boost;
            this.curvature = curvature;
        }

        /**
         * Evaluation value.
         */

        public double value;

        /**
         * Normal parameters.
         */

        public String evaluationMeasureValue;
        public String averagingStrategyValue;
        public boolean predictZeroRulesValue;
        public boolean readdAllCoveredValue;

        public double remainingInstancesPercentage;
        public double skipThresholdPercentage;

        /**
         * Boost function parameters.
         */

        public boolean useRelaxedPruning;

        public String boostFunctionValue;

        public int labelValue;
        public double boost;

        public double curvature;

        public void writeToCSV() {
            String[] headerRecord = {"Info", "Value"};
            csvWriter.writeNext(headerRecord);
            csvWriter.writeNext(new String[]{"baselearner", evaluationMeasureValue});
            csvWriter.writeNext(new String[]{"remainingInstancesPercentage", Double.toString(remainingInstancesPercentage)});
            csvWriter.writeNext(new String[]{"readdAllCovered", Boolean.toString(readdAllCoveredValue)});
            csvWriter.writeNext(new String[]{"skipThresholdPercentage", Double.toString(skipThresholdPercentage)});
            csvWriter.writeNext(new String[]{"predictZeroRules", Boolean.toString(predictZeroRulesValue)});
            csvWriter.writeNext(new String[]{"averagingStrategy", averagingStrategyValue});
            csvWriter.writeNext(new String[]{"useRelaxedPruning ", Boolean.toString(useRelaxedPruning)});
            csvWriter.writeNext(new String[]{"boostFunction", boostFunctionValue});
            csvWriter.writeNext(new String[]{"label", Double.toString(labelValue)});
            csvWriter.writeNext(new String[]{"boostAtLabel", Double.toString(boost)});
            csvWriter.writeNext(new String[]{"curvature", Double.toString(curvature)});
        }

    }

    /**
     * Normal parameters.
     */

    private static String[] evaluationMeasuresValues = new String[]{"f_measure", "hamming_accuracy", "subset_accuracy"};
    private static String[] averagingStrategyValues = new String[]{"micro-averaging", "macro-averaging"};
    private static boolean[] predictZeroRulesValues = new boolean[]{true, false};
    private static boolean[] readdAllCoveredValues = new boolean[]{true, false};

    private static double remainingInstancesMinimum = 0.0;
    private static double remainingInstancesMaximum = 0.9;
    private static double deltaRemainingInstances = 0.05;

    private static double skipThresholdMinimum = -0.01;
    private static double skipThresholdMaximum = 0.1;
    private static double deltaSkipThreshold = 0.02;

    /**
     * Boost function parameters.
     */

    private static boolean[] useRelaxedPruning = new boolean[]{true, false};

    private static String[] boostFunctionValues = new String[]{"peak", "root", "lln"};

    private static int labelValue = 2;

    private static double minimumBoost = 1.01;
    private static double maximumBoost = 1.3;
    private static double deltaBoost = 0.01;

    private static double minimumCurvature = 1.0;
    private static double maximumCurvature = 5.0;
    private static double deltaCurvature = 0.5;

    private static String getMandatoryArgument(final String argument, final String[] args) throws Exception {
        String value = Utils.getOption(argument, args);
        if (value.isEmpty())
            throw new IllegalArgumentException("Mandatory argument -" + argument + " missing");
        return value;
    }

    private static String getOptionalArgument(final String argument, final String[] args, final String defaultValue) throws Exception {
        String value = Utils.getOption(argument, args);
        if (value.isEmpty())
            return defaultValue;
        return value;
    }

    public static void mainEvaluation(final String[] args) throws Exception {

        String baseLearnerConfigPath = getMandatoryArgument("baselearner", args);
        String arffFilePath = getMandatoryArgument("arff", args);
        String xmlLabelsDefFilePath = getOptionalArgument("xml", args, arffFilePath.replace(".arff", ".xml"));
        String testArffFilePath = getOptionalArgument("test-arff", args, null);
        double remainingInstancesPercentage = Double.valueOf(getOptionalArgument("remainingInstancesPercentage", args, "0.05"));
        boolean readdAllCovered = Boolean.valueOf(getOptionalArgument("readdAllCovered", args, "false"));
        double skipThresholdPercentage = Double.valueOf(getOptionalArgument("skipThresholdPercentage", args, "-1.0"));
        boolean predictZeroRules = Boolean.valueOf(getOptionalArgument("predictZeroRules", args, "false"));
        boolean useMultilabelHeads = Boolean.valueOf(getOptionalArgument("useMultilabelHeads", args, "false"));
        String evaluationStrategy = getOptionalArgument("evaluationStrategy", args, EvaluationStrategy.RULE_DEPENDENT);
        String averagingStrategy = getOptionalArgument("averagingStrategy", args, AveragingStrategy.MICRO_AVERAGING);

        // relaxed pruning options
        boolean useRelaxedPruning = Boolean.valueOf((getOptionalArgument("useRelaxedPruning", args, "false")));
        boolean useBoostedHeuristicForRules = Boolean.valueOf(getOptionalArgument("useBoostedHeuristicForRules", args, "true"));
        String boostFunction = getOptionalArgument("boostFunction", args, "llm");
        double label = Double.valueOf(getOptionalArgument("label", args, "3.0"));
        double boostAtLabel = Double.valueOf(getOptionalArgument("boostAtLabel", args, "1.1"));
        double curvature = Double.valueOf(getOptionalArgument("curvature", args, "2.0"));
        int pruningDepth = Integer.valueOf(getOptionalArgument("pruningDepth", args, "-1"));

        // create training instances from dataset
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);

        System.out.println("SeCo: start EVALUATION experiment\n");

        // create test instances from dataset, if available
        final MultiLabelInstances testData = testArffFilePath != null ? new MultiLabelInstances(testArffFilePath, xmlLabelsDefFilePath) : null;

        for (String evaluationMeasureValue : evaluationMeasuresValues) {
            baseLearnerConfigPath = evaluationMeasureValue;
            for (String averagingStrategyValue : averagingStrategyValues) {
                averagingStrategy = averagingStrategyValue;

                Settings bestSetting = null;
                for (boolean predictZeroRulesValue : predictZeroRulesValues) {
                    predictZeroRules = predictZeroRulesValue;
                    for (boolean readdAllCoveredValue : readdAllCoveredValues) {
                        readdAllCovered = readdAllCoveredValue;
                        for (remainingInstancesPercentage = remainingInstancesMinimum; remainingInstancesPercentage <= remainingInstancesMaximum; remainingInstancesPercentage += deltaRemainingInstances) {
                            for (skipThresholdPercentage = skipThresholdMinimum; skipThresholdPercentage <= skipThresholdMaximum; skipThresholdPercentage += deltaSkipThreshold) {
                                if (!useRelaxedPruning) {

                                    SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile("config/" + baseLearnerConfigPath + ".xml");
                                    Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                                            remainingInstancesPercentage, readdAllCovered, skipThresholdPercentage, predictZeroRules,
                                            useMultilabelHeads, evaluationStrategy, averagingStrategy,
                                            useRelaxedPruning, useBoostedHeuristicForRules, boostFunction, label, boostAtLabel, curvature, pruningDepth);

                                    Evaluator evaluator = new Evaluator();
                                    MultipleEvaluation multipleEvaluation = evaluator.crossValidate(multilabelLearner, trainingData, 10);

                                    String measureName = getMeasureName(baseLearnerConfigPath, averagingStrategy);
                                    System.out.println(measureName);
                                    double value = multipleEvaluation.getMean(measureName);
                                    value = convertValue(baseLearnerConfigPath, value);
                                    Settings currentSetting = new Settings(value, baseLearnerConfigPath, averagingStrategy, predictZeroRules, readdAllCovered,
                                            remainingInstancesPercentage, skipThresholdPercentage);

                                    if (bestSetting == null || currentSetting.value > bestSetting.value) {
                                        bestSetting = currentSetting;
                                        continue;
                                    }

                                } else {

                                    for (String boostFunctionValue : boostFunctionValues) {
                                        boostFunction = boostFunctionValue;

                                        if (boostFunction.equalsIgnoreCase("peak")) {
                                            for (label = 2.0; label < trainingData.getNumLabels(); label++) {
                                                for (boostAtLabel = minimumBoost; boostAtLabel <= maximumBoost; boostAtLabel += deltaBoost) {
                                                    for (curvature = minimumCurvature; curvature <= maximumCurvature; curvature += deltaCurvature) {
                                                        SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile("config/" + baseLearnerConfigPath + ".xml");
                                                        Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                                                                remainingInstancesPercentage, readdAllCovered, skipThresholdPercentage, predictZeroRules,
                                                                useMultilabelHeads, evaluationStrategy, averagingStrategy,
                                                                useRelaxedPruning, useBoostedHeuristicForRules, boostFunction, label, boostAtLabel, curvature, pruningDepth);

                                                        Evaluator evaluator = new Evaluator();
                                                        MultipleEvaluation multipleEvaluation = evaluator.crossValidate(multilabelLearner, trainingData, 10);

                                                        String measureName = getMeasureName(baseLearnerConfigPath, averagingStrategy);
                                                        System.out.println(measureName);
                                                        double value = multipleEvaluation.getMean(measureName);
                                                        value = convertValue(baseLearnerConfigPath, value);
                                                        Settings currentSetting = new Settings(value, baseLearnerConfigPath, averagingStrategy, predictZeroRules, readdAllCovered,
                                                                remainingInstancesPercentage, skipThresholdPercentage,boostFunctionValue, (int) label, boostAtLabel, curvature);

                                                        if (bestSetting == null || currentSetting.value > bestSetting.value) {
                                                            bestSetting = currentSetting;
                                                            continue;
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            label = 3.0;
                                            for (boostAtLabel = minimumBoost; boostAtLabel <= maximumBoost; boostAtLabel += deltaBoost) {
                                                SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile("config/" + baseLearnerConfigPath + ".xml");
                                                Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                                                        remainingInstancesPercentage, readdAllCovered, skipThresholdPercentage, predictZeroRules,
                                                        useMultilabelHeads, evaluationStrategy, averagingStrategy,
                                                        useRelaxedPruning, useBoostedHeuristicForRules, boostFunction, label, boostAtLabel, curvature, pruningDepth);

                                                Evaluator evaluator = new Evaluator();
                                                MultipleEvaluation multipleEvaluation = evaluator.crossValidate(multilabelLearner, trainingData, 10);

                                                String measureName = getMeasureName(baseLearnerConfigPath, averagingStrategy);
                                                System.out.println(measureName);
                                                double value = multipleEvaluation.getMean(measureName);
                                                value = convertValue(baseLearnerConfigPath, value);
                                                Settings currentSetting = new Settings(value, baseLearnerConfigPath, averagingStrategy, predictZeroRules, readdAllCovered,
                                                        remainingInstancesPercentage, skipThresholdPercentage,boostFunctionValue, (int) label, boostAtLabel, curvature);

                                                if (bestSetting == null || currentSetting.value > bestSetting.value) {
                                                    bestSetting = currentSetting;
                                                    continue;
                                                }
                                            }
                                        }

                                    }

                                }
                            }
                        }
                    }
                }

                // create csv file
                SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
                filename = "results/experiments/" + xmlLabelsDefFilePath.split("/")[1].split("\\.")[0] + "_" + baseLearnerConfigPath + "_" + evaluationStrategy + "_" + sdf.format(new Date()) + ".csv";
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

                Main.csvWriter = csvWriter;

                bestSetting.writeToCSV();

                if (!useRelaxedPruning) {
                    SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile("config/" + bestSetting.evaluationMeasureValue + ".xml");
                    Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                            bestSetting.remainingInstancesPercentage, bestSetting.readdAllCoveredValue, bestSetting.skipThresholdPercentage, bestSetting.predictZeroRulesValue,
                            useMultilabelHeads, evaluationStrategy, bestSetting.averagingStrategyValue,
                            useRelaxedPruning, useBoostedHeuristicForRules, boostFunction, label, boostAtLabel, curvature, pruningDepth);


                    Evaluator evaluator = new Evaluator();
                    Evaluation evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
                    for (Measure measure : evaluation.getMeasures()) {
                        csvWriter.writeNext(new String[]{measure.getName(), Double.toString(measure.getValue())});
                    }
                }

                csvWriter.close();
            }
        }

    }

    public static String getMeasureName(String evaluationMeasure, String averagingStrategy) {
        String prefix = getAveragingPrefix(averagingStrategy);
        String suffix = getMeasureSuffix(evaluationMeasure);
        return prefix + " " + suffix;
    }

    public static String getAveragingPrefix(String averagingStrategy) {
        if (averagingStrategy.equalsIgnoreCase("micro-averaging"))
            return "Micro-averaged";
        if (averagingStrategy.equalsIgnoreCase("macro-averaging"))
            return "Macro-averaged";
        return null;
    }

    public static String getMeasureSuffix(String evaluationMeasure) {
        if (evaluationMeasure.equalsIgnoreCase("hamming_accuracy"))
            return "Hamming Loss";
        if (evaluationMeasure.equalsIgnoreCase("subset_accuracy"))
            return "Subset Accuracy";
        if (evaluationMeasure.equalsIgnoreCase("f_measure"))
            return "F-Measure";
        return null;
    }

    public static double convertValue(String evaluationMeasure, double value) {
        if (evaluationMeasure.equalsIgnoreCase("hamming_accuracy"))
            return (1.0 - value);
        return value;
    }

    public static CSVWriter csvWriter;
    public static FileWriter fileWriter;
    public static String filename;


}
