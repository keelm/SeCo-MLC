package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting.*;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting.KLNLifting;

import java.util.HashMap;

public abstract class LiftingStrategy {

    /**
     * Maximum number of labels a head can consist of.
     */
    protected double maximumNumberOfLabels;

    /**
     * Contains a lift function value for every possible head length.
     */
    protected HashMap<Integer, Double> liftFunctionValues = new HashMap<>();

    public LiftingStrategy(int maximumNumberOfLabels) {
        this.maximumNumberOfLabels = maximumNumberOfLabels;

    }

    /**
     * Creates the desired boosting strategy given parameters.
     */
    public static LiftingStrategy create(int maximumNumberOfLabels, String boostFunction, double label, double boostAtLabel, double curvature) {
        if (boostFunction.equalsIgnoreCase("peak")) {
            return new MaxLifting(maximumNumberOfLabels, label, boostAtLabel, curvature);
        } else if (boostFunction.equalsIgnoreCase("root")){
            return new RootLifting(maximumNumberOfLabels, label, boostAtLabel);
        } else {
            return new KLNLifting(maximumNumberOfLabels, label, boostAtLabel);
        }
    }

    /**
     * Evaluates the lift function for all possible values.
     */
    protected void evaluateForAllHeadLengths() {
        for (int headLength = 1; headLength <= maximumNumberOfLabels; headLength++) {
            double boostFunctionValue = lift(headLength);
            liftFunctionValues.put(headLength, boostFunctionValue);
        }
    }

    /**
     * Gets the maximum value of the lift function for the next evaluations.
     * @param headSize The current head size of the rule. Beginning of the interval.
     * @param lookahead How many further head sizes are taken into account for determining the maximum.
     * @return The maximum value of the lift function in [headSize, headSize + lookahead].
     */
    public double getMaximumLookaheadLift(int headSize, int lookahead) {
        double maximumBoostFunctionValue = -1;
        for (int headLength = headSize + 1; headLength <= headSize + lookahead && headLength <= maximumNumberOfLabels; headLength++) {
            double boostFunctionValue = liftFunctionValues.get(headLength);
            if (boostFunctionValue > maximumBoostFunctionValue)
                maximumBoostFunctionValue = boostFunctionValue;
        }
        return maximumBoostFunctionValue;
    }

    /**
     * Gets the maximum possible value of the lift function.
     * @param headSize The current head size of the rule.
     * @return The maximum lift function value.
     */
    public double getMaximumLift(int headSize) {
        double maximumBoostFunctionValue = -1;
        for (int headLength = headSize + 1; headLength <= maximumNumberOfLabels; headLength++) {
            double boostFunctionValue = liftFunctionValues.get(headLength);
            if (boostFunctionValue > maximumBoostFunctionValue)
                maximumBoostFunctionValue = boostFunctionValue;
        }
        return maximumBoostFunctionValue;
    }

    /**
     * The lift function.
     */
    protected abstract double lift(double x);

    /**
     * Applies the lift function to the rule. Sets the boosted heuristic
     * @param rule The rule the lift function is applied to.
     */
    public void evaluate(MultiHeadRule rule) {
        int numberOfLabelsInTheHead = rule.getHead().size();
        double rawRuleValue = rule.getRawRuleValue();
        double liftedRuleValue = applyLift(rawRuleValue, numberOfLabelsInTheHead);
        rule.setLiftedRuleValue(liftedRuleValue);
    }

    private double applyLift(double rawRuleValue, int numberOfLabelsInTheHead) {
        double lift = liftFunctionValues.get(numberOfLabelsInTheHead);
        double value = rawRuleValue * lift;
        return value;
    }

    public abstract String toString();

}
