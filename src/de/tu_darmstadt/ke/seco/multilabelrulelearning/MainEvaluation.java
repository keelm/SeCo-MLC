package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import com.opencsv.*;
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
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainEvaluation {

    private static class EvaluationSetting {

        public EvaluationSetting(double hamming, double subset, double value, String evaluationMeasureValue, String averagingStrategyValue, boolean predictZeroRulesValue,
                                 boolean readdAllCoveredValue, double remainingInstancesPercentage, double skipThresholdPercentage) {
            this.hamming = hamming;
            this.subset = subset;
            this.value = value;
            this.evaluationMeasureValue = evaluationMeasureValue;
            this.averagingStrategyValue = averagingStrategyValue;
            this.predictZeroRulesValue = predictZeroRulesValue;
            this.readdAllCoveredValue = readdAllCoveredValue;
            this.remainingInstancesPercentage = remainingInstancesPercentage;
            this.skipThresholdPercentage = skipThresholdPercentage;
            this.useRelaxedPruning = false;
            this.boostFunctionValue = "peak";
            this.labelValue = 3;
            this.boost = 1.1;
            this.curvature = 2.0;
        }

        public EvaluationSetting(double hamming, double subset, double value, String evaluationMeasureValue, String averagingStrategyValue, boolean predictZeroRulesValue,
                                 boolean readdAllCoveredValue, double remainingInstancesPercentage, double skipThresholdPercentage,
                                 String boostFunctionValue, int labelValue, double boost, double curvature) {
            this.hamming = hamming;
            this.subset = subset;
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

        public double hamming;
        public double subset;

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

        public void string(long id) {
            System.out.println(id + " baselearner " + evaluationMeasureValue);
            System.out.println(id + " remainingInstancesPercentage " + Double.toString(remainingInstancesPercentage));
            System.out.println(id + " readdAllCovered " + Boolean.toString(readdAllCoveredValue));
            System.out.println(id + " skipThresholdPercentage " + Double.toString(skipThresholdPercentage));
        System.out.println(id + " predictZeroRules " + Boolean.toString(predictZeroRulesValue));
        System.out.println(id + " averagingStrategy " + averagingStrategyValue);
        System.out.println(id + " useRelaxedPruning " + Boolean.toString(useRelaxedPruning));
        System.out.println(id + " boostFunction " + boostFunctionValue);
        System.out.println(id + " label " + Double.toString(labelValue));
        System.out.println(id + " boostAtLabel " + Double.toString(boost));
        System.out.println(id + " curvature " + Double.toString(curvature));
        }

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
    private static boolean[] readdAllCoveredValues = new boolean[]{true};

    private static double remainingInstancesMinimum = 0.1;
    private static double remainingInstancesMaximum = 0.1;
    private static double deltaRemainingInstances = 0.1;

    private static double skipThresholdMinimum = 0.01;
    private static double skipThresholdMaximum = 0.01;
    private static double deltaSkipThreshold = 0.02;

    /**
     * Boost function parameters.
     */

    private static boolean[] useRelaxedPruning = new boolean[]{true, false};

    private static String[] boostFunctionValues = new String[]{"peak", "lln"};

    private static int labelValue = 2;

    // TODO: floating point precision -> use list/array
    private static double minimumBoost = 1.01;
    private static double maximumBoost = 1.21;
    private static double deltaBoost = 0.02;

    private static double minimumCurvature = 2.0;
    private static double maximumCurvature = 2.0;
    private static double deltaCurvature = 1.0;

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

    private static ArrayList<EvaluationSetting> tasks = new ArrayList<>();

    private static int finished = 0;

    public static synchronized void finished() {
        finished++;
        System.out.println("finished " + finished);
    }

    public static void createTasks(final String[] args) throws Exception {
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
        String boostFunction = getOptionalArgument("boostFunction", args, "kln");
        double label = Double.valueOf(getOptionalArgument("label", args, "3.0"));
        double boostAtLabel = Double.valueOf(getOptionalArgument("boostAtLabel", args, "1.1"));
        double curvature = Double.valueOf(getOptionalArgument("curvature", args, "2.0"));
        int pruningDepth = Integer.valueOf(getOptionalArgument("pruningDepth", args, "-1"));

        // create training instances from dataset
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);
        // create test instances from dataset, if available
        final MultiLabelInstances testData = testArffFilePath != null ? new MultiLabelInstances(testArffFilePath, xmlLabelsDefFilePath) : null;

        System.out.println("Creating tasks...");
        /*for (boolean predictZeroRulesValue : predictZeroRulesValues) {
            predictZeroRules = predictZeroRulesValue;*/
        for (boolean readdAllCoveredValue : readdAllCoveredValues) {
            readdAllCovered = readdAllCoveredValue;
            for (remainingInstancesPercentage = remainingInstancesMinimum; remainingInstancesPercentage <= remainingInstancesMaximum; remainingInstancesPercentage += deltaRemainingInstances) {
                for (skipThresholdPercentage = skipThresholdMinimum; skipThresholdPercentage <= skipThresholdMaximum; skipThresholdPercentage += deltaSkipThreshold) {
                    if (!useRelaxedPruning) {
                        EvaluationSetting setting = new EvaluationSetting(-1, -1, -1, baseLearnerConfigPath, averagingStrategy, predictZeroRules,
                                readdAllCovered, remainingInstancesPercentage, skipThresholdPercentage);
                        tasks.add(setting);
                    } else {
                        for (String boostFunctionValue : boostFunctionValues) {
                            boostFunction = boostFunctionValue;

                            if (boostFunction.equalsIgnoreCase("peak")) {
                                for (label = 2.0; label < trainingData.getLabelIndices().length && label <= 5.0; label++) {
                                    if (label == 4.0)
                                        continue;
                                    for (boostAtLabel = minimumBoost; boostAtLabel <= maximumBoost; boostAtLabel += deltaBoost) {
                                        for (curvature = minimumCurvature; curvature <= maximumCurvature; curvature += deltaCurvature) {
                                            EvaluationSetting setting = new EvaluationSetting(-1, -1, -1, baseLearnerConfigPath, averagingStrategy, predictZeroRules, readdAllCovered,
                                                    remainingInstancesPercentage, skipThresholdPercentage, boostFunctionValue, (int) label, boostAtLabel, curvature);
                                            tasks.add(setting);
                                        }
                                    }
                                }
                            } else {
                                label = 3.0;
                                for (boostAtLabel = minimumBoost; boostAtLabel <= maximumBoost; boostAtLabel += deltaBoost) {
                                    EvaluationSetting setting = new EvaluationSetting(-1, -1, -1, baseLearnerConfigPath, averagingStrategy, predictZeroRules, readdAllCovered,
                                            remainingInstancesPercentage, skipThresholdPercentage, boostFunctionValue, (int) label, boostAtLabel, curvature);
                                    tasks.add(setting);
                                }
                            }
                        }
                    }
                }
            }
        }
        //}
        System.out.println("Created " + tasks.size() + " tasks...");

        ExecutorService executorService = Executors.newFixedThreadPool(NUMBER_OF_THREADS);
        for (int i = 0; i < NUMBER_OF_THREADS; i++) {
            executorService.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        executeTasks(baseLearnerConfigPath, trainingData, testData);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });
        }

        while (finished != NUMBER_OF_THREADS) {
            System.out.println("Checking if all finished...");
            Thread.sleep(1000*60);
        }

        System.out.println("Found Best Setting " + bestSetting.value);

        executorService.shutdown();

        // create csv file
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss.SSS");
        filename = "results/experiments/" + xmlLabelsDefFilePath.split("/")[1].split("\\.")[0] + "_" + getMeasureName(bestSetting.evaluationMeasureValue, bestSetting.averagingStrategyValue) + "_" + sdf.format(new Date()) + ".csv";
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

        bestSetting.string(1);

       /* if (!useRelaxedPruning) {
            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                    bestSetting.remainingInstancesPercentage, bestSetting.readdAllCoveredValue, bestSetting.skipThresholdPercentage, bestSetting.predictZeroRulesValue,
                    useMultilabelHeads, evaluationStrategy, bestSetting.averagingStrategyValue,
                    useRelaxedPruning, useBoostedHeuristicForRules, boostFunction, label, boostAtLabel, curvature, pruningDepth);


            Evaluator evaluator = new Evaluator();
            multilabelLearner.build(trainingData);
            Evaluation evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
            for (Measure measure : evaluation.getMeasures()) {
                csvWriter.writeNext(new String[]{measure.getName(), Double.toString(measure.getValue())});
            }
        } else {
            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                    bestSetting.remainingInstancesPercentage, bestSetting.readdAllCoveredValue, bestSetting.skipThresholdPercentage, bestSetting.predictZeroRulesValue,
                    useMultilabelHeads, evaluationStrategy, bestSetting.averagingStrategyValue,
                    useRelaxedPruning, useBoostedHeuristicForRules, bestSetting.boostFunctionValue, bestSetting.labelValue, bestSetting.lift, bestSetting.curvature, -1);

            Evaluator evaluator = new Evaluator();
            multilabelLearner.build(trainingData);
            Evaluation evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
            for (Measure measure : evaluation.getMeasures()) {
                csvWriter.writeNext(new String[]{measure.getName(), Double.toString(measure.getValue())});
            }
        }*/

        csvWriter.close();
        System.out.println("done");

    }

    private static int NUMBER_OF_THREADS = 40;

    public synchronized static EvaluationSetting getSetting() {
        if (tasks.isEmpty())
            return null;
        EvaluationSetting setting = tasks.get(0);
        tasks.remove(0);
        System.out.println("Remaining Tasks: " + tasks.size());
        return setting;
    }

    private static final int CROSS_VALIDATION_FOLDS = 5;

    public static void executeTasks(final String baseLearnerConfigPath, final MultiLabelInstances trainingData, final MultiLabelInstances testData) throws Exception {
        while (!tasks.isEmpty()) {
            EvaluationSetting setting = getSetting();
            if (setting == null || setting.value != -1)
                continue;

            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                    setting.remainingInstancesPercentage, setting.readdAllCoveredValue, setting.skipThresholdPercentage, setting.predictZeroRulesValue,
                    true, EvaluationStrategy.RULE_DEPENDENT, setting.averagingStrategyValue,
                    setting.useRelaxedPruning, true, setting.boostFunctionValue, setting.labelValue, setting.boost, setting.curvature, -1, true);

            Evaluator evaluator = new Evaluator();
            MultipleEvaluation multipleEvaluation = evaluator.crossValidate(multilabelLearner, trainingData, CROSS_VALIDATION_FOLDS);

            String measureName = getMeasureName(baseLearnerConfigPath, setting.averagingStrategyValue);
            double value = multipleEvaluation.getMean(measureName);
            value = convertValue(setting.evaluationMeasureValue, value);
            setting.value = value;

            double hamming = multipleEvaluation.getMean("Hamming Loss");
            hamming = convertValue("config/hamming_accuracy.xml", hamming);
            setting.hamming = hamming;

            double subset = multipleEvaluation.getMean("Subset Accuracy");
            setting.subset = subset;

            updateBestSetting(setting);
        }
        finished();
    }

    private static EvaluationSetting bestSetting = null;

    public synchronized static void updateBestSetting(EvaluationSetting setting) {
        if (bestSetting != null) {
            if (setting.useRelaxedPruning)
                System.out.println("Updating Best Setting " + bestSetting.value + "," + bestSetting.hamming + "," + bestSetting.subset + "," + bestSetting.boost + " ("+ setting.value + "," + setting.hamming + "," + setting.subset + "," + setting.boost + ")");
            else
                System.out.println("Updating Best Setting " + bestSetting.value + "," + bestSetting.hamming + "," + bestSetting.subset + " ("+ setting.value + "," + setting.hamming + "," + setting.subset + ")");
        } if (bestSetting == null || setting.value >= bestSetting.value) {
            if (bestSetting != null && setting.value == bestSetting.value) {
                if (setting.hamming >= bestSetting.hamming) {
                    if (setting.hamming == bestSetting.hamming) {
                        if (setting.subset >= bestSetting.subset) {
                            if (setting.subset == bestSetting.subset) {
                                if (setting.useRelaxedPruning && setting.boost < bestSetting.boost) {
                                    bestSetting = setting;
                                }
                            } else {
                                bestSetting = setting;
                            }
                        }
                    } else {
                        bestSetting = setting;
                    }
                }
            } else {
                bestSetting = setting;
            }
        }
    }

    public static String getMeasureName(String evaluationMeasure, String averagingStrategy) {
        String prefix = getAveragingPrefix(averagingStrategy);
        String suffix = getMeasureSuffix(evaluationMeasure);
        return needsPrefix(evaluationMeasure) ? prefix + " " + suffix : suffix;
    }

    public static boolean needsPrefix(String evaluationMeasure) {
        if (evaluationMeasure.equalsIgnoreCase("config/hamming_accuracy.xml"))
            return false;
        if (evaluationMeasure.equalsIgnoreCase("config/subset_accuracy.xml"))
            return false;
        if (evaluationMeasure.equalsIgnoreCase("config/f_measure.xml"))
            return true;
        return false;
    }

    public static String getAveragingPrefix(String averagingStrategy) {
        if (averagingStrategy.equalsIgnoreCase("micro-averaging"))
            return "Micro-averaged";
        if (averagingStrategy.equalsIgnoreCase("macro-averaging"))
            return "Macro-averaged";
        return null;
    }

    public static String getMeasureSuffix(String evaluationMeasure) {
        if (evaluationMeasure.equalsIgnoreCase("config/hamming_accuracy.xml"))
            return "Hamming Loss";
        if (evaluationMeasure.equalsIgnoreCase("config/subset_accuracy.xml"))
            return "Subset Accuracy";
        if (evaluationMeasure.equalsIgnoreCase("config/f_measure.xml"))
            return "F-Measure";
        return null;
    }

    public static double convertValue(String evaluationMeasure, double value) {
        if (evaluationMeasure.equalsIgnoreCase("config/hamming_accuracy.xml"))
            return (1.0 - value);
        return value;
    }

    public static CSVWriter csvWriter;
    public static FileWriter fileWriter;
    public static String filename;

}