package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class MaxBoosting extends BoostingStrategy {

    /**
     * Number of labels at which the boost function is maximal.
     */
    private double maximum;

    /**
     * Boost function value at the maximum.
     */
    private double boostAtMaximum;

    /**
     * Curvature of the function while still abiding (1,1) and (maximum, boostAtMaximum).
     * A curvature of 1.0 corresponds to a linear function.
     */
    private double curvature;

    /**
     * Maximum number of labels a head can consist of.
     */
    private double maxNumberOfLabels = 50.0; // TODO: set automatically

    private double numberOfLabelsInTheHead;

    private double rawRuleValue;
    private double boostedRuleValue;

    public MaxBoosting(double maximum, double boostAtMaximum, double curvature) {
        this.maximum = maximum;
        this.boostAtMaximum = boostAtMaximum;
        this.curvature = curvature;
    }

    @Override
    public void evaluate(MultiHeadRule rule) {
        numberOfLabelsInTheHead = rule.getHead().size();
        rawRuleValue = rule.getRawRuleValue();
        boostedRuleValue = boost();
        rule.setBoostedRuleValue(boostedRuleValue);
    }

    @Override
    public double evaluate(MultiHeadRule rule, double numberOfLabelsInTheHead) {
        this.numberOfLabelsInTheHead = numberOfLabelsInTheHead;
        rawRuleValue = rule.getRawRuleValue();
        boostedRuleValue = boost();
        return boostedRuleValue;
    }

    private double boost() {
        double boost = boost(numberOfLabelsInTheHead);
        double value = rawRuleValue * boost;
        return value;
    }

    private double boost(double x) {
        if (x <= maximum)
            return boostBeforeMaximum(x);
        return boostAfterMaximum(x);
    }

    private double boostBeforeMaximum(double x) {
        double exponent = 1.0 / curvature;
        double normalization = (x - 1.0) / (maximum - 1.0);
        double boost = 1.0 + Math.pow(normalization, exponent) * (boostAtMaximum - 1.0);
        return boost;
    }

    private double boostAfterMaximum(double x) {
        double exponent = 1.0 / curvature;
        double normalization = (x - maxNumberOfLabels) / (maxNumberOfLabels - maximum);
        double boost = 1.0 + Math.pow(-normalization, exponent) * (boostAtMaximum - 1.0);
        return boost;
    }

    public String toString() {
        return "MaxBoosting(" + maximum + "|" + boostAtMaximum + " | " + curvature + ")";
    }



}
