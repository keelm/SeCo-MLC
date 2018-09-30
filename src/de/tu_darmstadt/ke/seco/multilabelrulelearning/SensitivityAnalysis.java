package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import de.tu_darmstadt.ke.seco.Main;
import de.tu_darmstadt.ke.seco.models.MultiHeadRuleSet;

public class SensitivityAnalysis {

    private static String[] boostFunctionValues = new String[]{"peak", "lln"};

    public static void main(String args[]) {

        for (String boostFunction : boostFunctionValues) {
            if (boostFunction == "lln") {
                String label = "3.0";
                for (double b = 1.01; b <= 1.30; b += 0.02) {
                    String boostAtLabel = String.valueOf(b);
                    String[] arguments = new String[]{"-baselearner", "config/f_measure.xml", "-arff", "data/flags-train.arff", "-xml", "data/flags.xml",
                            "-test-arff", "data/flags-test.arff", "-remainingInstancesPercentage", "0.1", "-readdAllCovered", "true",
                            "-skipThresholdPercentage", "0.01", "-predictZeroRules", "true", "-useMultilabelHeads", "true", "-averagingStrategy",
                            "micro-averaging", "-evaluationStrategy", "rule-dependent", "-useRelaxedPruning", "true", "-useBoostedHeuristicForRules",
                            "true", "-boostFunction", boostFunction, "-label", label, "-boostAtLabel", boostAtLabel, "-curvature", "2.0", "-pruningDepth", "-1"};
                    try {
                        Main.main(arguments);
                        MulticlassCovering.finished = false;
                        MultiHeadRuleSet.csvWritten = false;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            } else {
                for (double l = 2.0; l <= 5.0; l++) {
                    String label = String.valueOf(l);
                    for (double b = 1.01; b <= 1.30; b += 0.02) {
                        String boostAtLabel = String.valueOf(b);
                        String[] arguments = new String[]{"-baselearner", "config/f_measure.xml", "-arff", "data/flags-train.arff", "-xml", "data/flags.xml",
                                "-test-arff", "data/flags-test.arff", "-remainingInstancesPercentage", "0.1", "-readdAllCovered", "true",
                                "-skipThresholdPercentage", "0.01", "-predictZeroRules", "true", "-useMultilabelHeads", "true", "-averagingStrategy",
                                "micro-averaging", "-evaluationStrategy", "rule-dependent", "-useRelaxedPruning", "true", "-useBoostedHeuristicForRules",
                                "true", "-boostFunction", boostFunction, "-label", label, "-boostAtLabel", boostAtLabel, "-curvature", "2.0", "-pruningDepth", "-1"};
                        try {
                            Main.main(arguments);
                            MulticlassCovering.finished = false;
                            MultiHeadRuleSet.csvWritten = false;
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }

        }

    }

}
