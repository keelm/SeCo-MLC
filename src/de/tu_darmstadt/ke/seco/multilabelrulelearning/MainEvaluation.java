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

import static de.tu_darmstadt.ke.seco.multilabelrulelearning.MulticlassCovering.evaluatedHeads;
import static de.tu_darmstadt.ke.seco.multilabelrulelearning.MulticlassCovering.evaluationsPerHead;

public class MainEvaluation {

    /** The number of threads used for executing the tasks. */
    private static int NUMBER_OF_THREADS = 4;

    /** The number of cross validations folds used for estimating the performance on the training set. */
    private static final int CROSS_VALIDATION_FOLDS = 5;

    /** Normal parameters. */
    private static boolean[] reAddAllCoveredValues = {true};
    private static double[] remainingInstancesPercentageValues = {0.1};
    private static double[] skipThresholdPercentageValues = {0.01};

    /** Relaxation lift function parameters. */
    private static String[] liftFunctions = new String[]{"peak", "kln"};
    private static int[] labelValues = {2, 3, 5}; // values MUST be ordered increasingly
    private static double[] liftValues = {1.00, 1.02, 1.04, 1.08, 1.12, 1.16, 1.20, 1.24};
    private static double[] curvatures = {2.0};


    /** Class for storing a parameter set. */
    private static class EvaluationSetting {

        public EvaluationSetting(double hamming, double subset, double value, String evaluationMeasureValue, String averagingStrategyValue, boolean predictZeroRulesValue,
                                 boolean reAddAllCoveredValue, double remainingInstancesPercentage, double skipThresholdPercentage) {
            this.hamming = hamming;
            this.subset = subset;
            this.value = value;
            this.evaluationMeasureValue = evaluationMeasureValue;
            this.averagingStrategyValue = averagingStrategyValue;
            this.predictZeroRulesValue = predictZeroRulesValue;
            this.reAddAllCoveredValue = reAddAllCoveredValue;
            this.remainingInstancesPercentage = remainingInstancesPercentage;
            this.skipThresholdPercentage = skipThresholdPercentage;
            this.useRelaxedPruning = false;
            this.liftFunction = "peak";
            this.labelValue = 3;
            this.lift = 1.1;
            this.curvature = 2.0;
        }

        public EvaluationSetting(double hamming, double subset, double value, String evaluationMeasureValue, String averagingStrategyValue, boolean predictZeroRulesValue,
                                 boolean reAddAllCoveredValue, double remainingInstancesPercentage, double skipThresholdPercentage,
                                 String liftFunction, int labelValue, double lift, double curvature) {
            this.hamming = hamming;
            this.subset = subset;
            this.value = value;
            this.evaluationMeasureValue = evaluationMeasureValue;
            this.averagingStrategyValue = averagingStrategyValue;
            this.predictZeroRulesValue = predictZeroRulesValue;
            this.reAddAllCoveredValue = reAddAllCoveredValue;
            this.remainingInstancesPercentage = remainingInstancesPercentage;
            this.skipThresholdPercentage = skipThresholdPercentage;
            this.useRelaxedPruning = true;
            this.liftFunction = liftFunction;
            this.labelValue = labelValue;
            this.lift = lift;
            this.curvature = curvature;
        }


        /** Evaluation value. */
        public double value;

        public double hamming;
        public double subset;

        /** Normal parameters. */
        public String evaluationMeasureValue;
        public String averagingStrategyValue;
        public boolean predictZeroRulesValue;
        public boolean reAddAllCoveredValue;
        public double remainingInstancesPercentage;
        public double skipThresholdPercentage;

        /** Relaxation lift function parameters. */
        public boolean useRelaxedPruning;
        public String liftFunction;
        public int labelValue;
        public double lift;
        public double curvature;


        public void string(long id) {
            System.out.println(id + " baselearner " + evaluationMeasureValue);
            System.out.println(id + " remainingInstancesPercentage " + Double.toString(remainingInstancesPercentage));
            System.out.println(id + " reAddAllCovered " + Boolean.toString(reAddAllCoveredValue));
            System.out.println(id + " skipThresholdPercentage " + Double.toString(skipThresholdPercentage));
            System.out.println(id + " predictZeroRules " + Boolean.toString(predictZeroRulesValue));
            System.out.println(id + " averagingStrategy " + averagingStrategyValue);
            System.out.println(id + " useRelaxedPruning " + Boolean.toString(useRelaxedPruning));
            System.out.println(id + " liftFunction " + liftFunction);
            System.out.println(id + " label " + Double.toString(labelValue));
            System.out.println(id + " liftAtLabel " + Double.toString(lift));
            System.out.println(id + " curvature " + Double.toString(curvature));
        }

        public void writeToCSV() {
            String[] headerRecord = {"Info", "Value"};
            csvWriter.writeNext(headerRecord);
            csvWriter.writeNext(new String[]{"baselearner", evaluationMeasureValue});
            csvWriter.writeNext(new String[]{"remainingInstancesPercentage", Double.toString(remainingInstancesPercentage)});
            csvWriter.writeNext(new String[]{"reAddAllCovered", Boolean.toString(reAddAllCoveredValue)});
            csvWriter.writeNext(new String[]{"skipThresholdPercentage", Double.toString(skipThresholdPercentage)});
            csvWriter.writeNext(new String[]{"predictZeroRules", Boolean.toString(predictZeroRulesValue)});
            csvWriter.writeNext(new String[]{"averagingStrategy", averagingStrategyValue});
            csvWriter.writeNext(new String[]{"useRelaxedPruning ", Boolean.toString(useRelaxedPruning)});
            csvWriter.writeNext(new String[]{"liftFunction", liftFunction});
            csvWriter.writeNext(new String[]{"label", Double.toString(labelValue)});
            csvWriter.writeNext(new String[]{"liftAtLabel", Double.toString(lift)});
            csvWriter.writeNext(new String[]{"curvature", Double.toString(curvature)});
        }

    }

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


    /** A list of all remaining tasks. */
    private static ArrayList<EvaluationSetting> tasks = new ArrayList<>();

    /** Number of finished threads. */
    private static int finished = 0;

    /** Called by a thread upon termination.*/
    public static synchronized void finished() {
        finished++;
        System.out.println("finished " + finished);
    }


    /** Creates all sets of parameters that need to be tested. */
    public static void createTasks(final String[] args) throws Exception {
        String baseLearnerConfigPath = getMandatoryArgument("baselearner", args);
        String arffFilePath = getMandatoryArgument("arff", args);
        String xmlLabelsDefFilePath = getOptionalArgument("xml", args, arffFilePath.replace(".arff", ".xml"));
        String testArffFilePath = getOptionalArgument("test-arff", args, null);
        boolean predictZeroRules = Boolean.valueOf(getOptionalArgument("predictZeroRules", args, "false"));
        String averagingStrategy = getOptionalArgument("averagingStrategy", args, AveragingStrategy.MICRO_AVERAGING);
        // relaxed pruning options
        boolean useRelaxedPruning = Boolean.valueOf((getOptionalArgument("useRelaxedPruning", args, "false")));

        // create training instances from data set
        final MultiLabelInstances trainingData = new MultiLabelInstances(arffFilePath, xmlLabelsDefFilePath);
        // create test instances from data set, if available
        final MultiLabelInstances testData = testArffFilePath != null ? new MultiLabelInstances(testArffFilePath, xmlLabelsDefFilePath) : null;

        // iterate through all specified combinations of parameters
        for (boolean reAddAllCovered : reAddAllCoveredValues) {
            for (double remainingInstancesPercentage : remainingInstancesPercentageValues) {
                for (double skipThresholdPercentage : skipThresholdPercentageValues) {
                    // use different setting if not using relaxed pruning
                    if (!useRelaxedPruning) {
                        EvaluationSetting setting = new EvaluationSetting(-1, -1, -1, baseLearnerConfigPath,
                                averagingStrategy, predictZeroRules, reAddAllCovered, remainingInstancesPercentage, skipThresholdPercentage);
                        tasks.add(setting);
                    } else {
                        // for every peak function value (here: only peak and kln)
                        for (String liftFunction : liftFunctions) {
                            if (liftFunction.equalsIgnoreCase("peak")) {
                                // for every peak label
                                for (int label : labelValues) {
                                    // assumes values are ordered increasingly
                                    if (label > trainingData.getLabelIndices().length)
                                        break;
                                    // for every specified extent of the lift
                                    for (double liftValue : liftValues) {
                                        // for every specified curvature
                                        for (double curvature : curvatures) {
                                            EvaluationSetting setting = new EvaluationSetting(-1, -1, -1, baseLearnerConfigPath,
                                                    averagingStrategy, predictZeroRules, reAddAllCovered, remainingInstancesPercentage,
                                                    skipThresholdPercentage, liftFunction, label, liftValue, curvature);
                                            tasks.add(setting);
                                        }
                                    }
                                }
                            } else {
                                int label = 3;
                                // for every specified extent of the lift
                                for (double liftValue : liftValues) {
                                    // for every specified curvature
                                    for (double curvature : curvatures) {
                                        EvaluationSetting setting = new EvaluationSetting(-1, -1, -1, baseLearnerConfigPath,
                                                averagingStrategy, predictZeroRules, reAddAllCovered, remainingInstancesPercentage,
                                                skipThresholdPercentage, liftFunction, label, liftValue, curvature);
                                        tasks.add(setting);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        System.out.println("created " + tasks.size() + " tasks");

        // start executing tasks on NUMBER_OF_THREADS threads
        ExecutorService executorService = Executors.newFixedThreadPool(NUMBER_OF_THREADS);
        for (int i = 0; i < NUMBER_OF_THREADS; i++) {
            executorService.execute(() -> {
                try {
                    executeTasks(baseLearnerConfigPath, trainingData, testData);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }

        // check every 60 seconds whether all threads are done
        while (finished != NUMBER_OF_THREADS) {
            System.out.println("checking whether all threads are done...");
            Thread.sleep(1000*60);
        }

        System.out.println("best setting: " + bestSetting.value);

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

        // reset values to get results of best parameter setting
        evaluationsPerHead = 0;
        evaluatedHeads = 0;

        // execute the best parameter set to get final evaluation result
        Weka379AdapterMultilabel multilabelLearner;
        if (!useRelaxedPruning) {
            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                    bestSetting.remainingInstancesPercentage, bestSetting.reAddAllCoveredValue, bestSetting.skipThresholdPercentage, bestSetting.predictZeroRulesValue,
                    true, EvaluationStrategy.RULE_DEPENDENT, bestSetting.averagingStrategyValue,
                    false, true, "peak", 1, 1.0, 1.0, -1, true); // relaxed pruning values are don't cares
        } else {
            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                    bestSetting.remainingInstancesPercentage, bestSetting.reAddAllCoveredValue, bestSetting.skipThresholdPercentage, bestSetting.predictZeroRulesValue,
                    true, EvaluationStrategy.RULE_DEPENDENT, bestSetting.averagingStrategyValue,
                    true, true, bestSetting.liftFunction, bestSetting.labelValue, bestSetting.lift, bestSetting.curvature, -1, true);
        }

        // build learner and evaluate on test set
        Evaluator evaluator = new Evaluator();
        multilabelLearner.build(trainingData);
        Evaluation evaluation = evaluator.evaluate(multilabelLearner, testData, trainingData);
        // write results to csv file
        for (Measure measure : evaluation.getMeasures())
            csvWriter.writeNext(new String[]{measure.getName(), Double.toString(measure.getValue())});

        csvWriter.writeNext(new String[]{"avg. #evals per findBestHead()", Double.toString(evaluationsPerHead)});
        csvWriter.writeNext(new String[]{"#findBestHead()", Integer.toString(evaluatedHeads)});
        // includes building time
        csvWriter.writeNext(new String[]{"model", multilabelLearner.toString().split("RuleSet")[1]});

        csvWriter.close();
        System.out.println("done");
    }


    /** Returns a set of parameters that still needs to be tested. Null if none exist. */
    public synchronized static EvaluationSetting getSetting() {
        if (tasks.isEmpty())
            return null;
        EvaluationSetting setting = tasks.get(0);
        tasks.remove(0);
        System.out.println("remaining tasks: " + tasks.size());
        return setting;
    }


    /** Executes a cross-validated experiment on a single parameter set. */
    public static void executeTasks(final String baseLearnerConfigPath, final MultiLabelInstances trainingData, final MultiLabelInstances testData) throws Exception {
        while (!tasks.isEmpty()) {
            // fetch new tasks
            EvaluationSetting setting = getSetting();
            if (setting == null || setting.value != -1)
                continue;

            SeCoAlgorithm baseLearnerAlgorithm = SeCoAlgorithmFactory.buildAlgorithmFromFile(baseLearnerConfigPath);
            Weka379AdapterMultilabel multilabelLearner = new Weka379AdapterMultilabel(baseLearnerAlgorithm,
                    setting.remainingInstancesPercentage, setting.reAddAllCoveredValue, setting.skipThresholdPercentage, setting.predictZeroRulesValue,
                    true, EvaluationStrategy.RULE_DEPENDENT, setting.averagingStrategyValue,
                    setting.useRelaxedPruning, true, setting.liftFunction, setting.labelValue, setting.lift, setting.curvature, -1, true);

            Evaluator evaluator = new Evaluator();
            MultipleEvaluation multipleEvaluation = evaluator.crossValidate(multilabelLearner, trainingData, CROSS_VALIDATION_FOLDS);

            // get evaluation metric to be optimized
            String measureName = getMeasureName(baseLearnerConfigPath, setting.averagingStrategyValue);
            double value = multipleEvaluation.getMean(measureName);
            value = convertValue(setting.evaluationMeasureValue, value);
            setting.value = value;
            // get hamming accuracy
            double hamming = multipleEvaluation.getMean("Hamming Loss");
            hamming = convertValue("config/hamming_accuracy.xml", hamming);
            setting.hamming = hamming;
            // get subset accuracy
            double subset = multipleEvaluation.getMean("Subset Accuracy");
            setting.subset = subset;

            updateBestSetting(setting);
        }
        finished();
    }


    /** The best parameter set so far. */
    private static EvaluationSetting bestSetting = null;

    /** Updates the best parameter set if given 'better' parameter set. */
    public synchronized static void updateBestSetting(EvaluationSetting setting) {
        // print best and current parameter set evaluation metrics
        if (bestSetting != null) {
            if (setting.useRelaxedPruning)
                System.out.println("updating best setting: \n\tbest(" + bestSetting.value + ", " + bestSetting.hamming + ", " + bestSetting.subset + " | " + bestSetting.liftFunction + ", "  + bestSetting.labelValue + ", "  + bestSetting.lift + ")" +
                        "\n\tcurrent("+ setting.value + ", " + setting.hamming + ", " + setting.subset + " | " + setting.liftFunction + ", "  + setting.labelValue + ", "  + setting.lift + ")");
            else
                System.out.println("updating best setting: best(" + bestSetting.value + "," + bestSetting.hamming + "," + bestSetting.subset + ")" +
                        " vs. current("+ setting.value + "," + setting.hamming + "," + setting.subset + ")");
        // check whether parameter set better
        }
        if (bestSetting == null || setting.value >= bestSetting.value) {
            // same evaluation measure to be optimized value
            if (bestSetting != null && setting.value == bestSetting.value) {
                if (setting.hamming >= bestSetting.hamming) {
                    // same hamming accuracy
                    if (setting.hamming == bestSetting.hamming) {
                        if (setting.subset >= bestSetting.subset) {
                            // same subset accuracy
                            if (setting.subset == bestSetting.subset) {
                                // only set to be better if lift setting is smaller! i.e. we prefer smaller lift settings
                                if (setting.useRelaxedPruning && setting.lift < bestSetting.lift)
                                    bestSetting = setting;
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


    /** Generates a valid argument for Evaluator result query. **/
    public static String getMeasureName(String evaluationMeasure, String averagingStrategy) {
        String prefix = getAveragingPrefix(averagingStrategy);
        String suffix = getMeasureSuffix(evaluationMeasure);
        return needsPrefix(evaluationMeasure) ? prefix + " " + suffix : suffix;
    }

    /** True if the averaging strategy is part of the measure name, false otherwise. */
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

    /** Converts hamming loss to hamming accuracy. */
    public static double convertValue(String evaluationMeasure, double value) {
        if (evaluationMeasure.equalsIgnoreCase("config/hamming_accuracy.xml"))
            return (1.0 - value);
        return value;
    }

    public static CSVWriter csvWriter;
    public static FileWriter fileWriter;
    public static String filename;

}