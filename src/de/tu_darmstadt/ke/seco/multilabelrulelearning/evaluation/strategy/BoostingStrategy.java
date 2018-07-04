package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting.LLNBoosting;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting.MaxBoosting;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting.RootBoosting;

import java.util.HashMap;

public abstract class BoostingStrategy {

    /**
     * Maximum number of labels a head can consist of.
     */
    protected double maximumNumberOfLabels;

    /**
     * Contains a boost function value for every possible head length.
     */
    protected HashMap<Integer, Double> boostFunctionValues = new HashMap<>();

    public BoostingStrategy(int maximumNumberOfLabels) {
        this.maximumNumberOfLabels = maximumNumberOfLabels;

    }

    /**
     * Creates the desired boosting strategy given parameters.
     */
    public static BoostingStrategy create(int maximumNumberOfLabels, String boostFunction, double label, double boostAtLabel, double curvature) {
        if (boostFunction.equalsIgnoreCase("peak")) {
            return new MaxBoosting(maximumNumberOfLabels, label, boostAtLabel, curvature);
        } else if (boostFunction.equalsIgnoreCase("root")){
            return new RootBoosting(maximumNumberOfLabels, label, boostAtLabel);
        } else {
            return new LLNBoosting(maximumNumberOfLabels, label, boostAtLabel);
        }
    }

    /**
     * Evaluates the boost function for all possible values.
     */
    protected void evaluateForAllHeadLengths() {
        for (int headLength = 1; headLength <= maximumNumberOfLabels; headLength++) {
            double boostFunctionValue = boost(headLength);
            boostFunctionValues.put(headLength, boostFunctionValue);
        }
    }

    /**
     * Gets the maximum value of the boost function for the next evaluations.
     * @param headSize The current head size of the rule. Beginning of the interval.
     * @param lookahead How many further head sizes are taken into account for determining the maximum.
     * @return The maximum value of the boost function in [headSize, headSize + lookahead].
     */
    public double getMaximumLookaheadBoost(int headSize, int lookahead) {
        double maximumBoostFunctionValue = Double.MIN_VALUE;
        for (int headLength = headSize; headLength <= headSize + lookahead && headLength <= maximumNumberOfLabels; headLength++) {
            double boostFunctionValue = boostFunctionValues.get(headLength);
            if (boostFunctionValue > maximumBoostFunctionValue)
                maximumBoostFunctionValue = boostFunctionValue;
        }
        return maximumBoostFunctionValue;
    }

    /**
     * Gets the maximum possible value of the boost function.
     * @param headSize The current head size of the rule.
     * @return The maximum boost function value.
     */
    public double getMaximumBoost(int headSize) {
        double maximumBoostFunctionValue = Double.MIN_VALUE;
        for (int headLength = headSize; headLength <= maximumNumberOfLabels; headLength++) {
            double boostFunctionValue = boostFunctionValues.get(headLength);
            if (boostFunctionValue > maximumBoostFunctionValue)
                maximumBoostFunctionValue = boostFunctionValue;
        }
        return maximumBoostFunctionValue;
    }

    /**
     * The boost function.
     */
    protected abstract double boost(double x);

    /**
     * Applies the boost function to the rule. Sets the boosted heuristic
     * @param rule The rule the boost function is applied to.
     */
    public void evaluate(MultiHeadRule rule) {
        int numberOfLabelsInTheHead = rule.getHead().size();
        double rawRuleValue = rule.getRawRuleValue();
        double boostedRuleValue = applyBoost(rawRuleValue, numberOfLabelsInTheHead);
        rule.setBoostedRuleValue(boostedRuleValue);
    }

    private double applyBoost(double rawRuleValue, int numberOfLabelsInTheHead) {
        double boost = boostFunctionValues.get(numberOfLabelsInTheHead);
        double value = rawRuleValue * boost;
        return value;
    }

    public abstract String toString();

}
