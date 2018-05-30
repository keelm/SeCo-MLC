package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class LogXBoosting extends BoostingStrategy {

    /**
     * Manual Settings.
     */
    private double a = 1.0; // gradient
    private double b = 4.2; // shifts maximum

    /**
     * Automatic Settings. Sets Parameters a and b according to the maximum and the wished boost at that maximum.
     */
    private boolean useAutomaticSettings = true;
    private double maximum = 3.0;
    private double boostAtMaximum = 1.5;

    private double numberOfLabelsInTheHead;

    private double rawRuleValue;
    private double boostedRuleValue;

    @Override
    public void evaluate(MultiHeadRule rule) {
        numberOfLabelsInTheHead = rule.getHead().size();
        rawRuleValue = rule.getRawRuleValue();
        if (useAutomaticSettings)
            setParameters();
        boostedRuleValue = logValue();
        rule.setBoostedRuleValue(boostedRuleValue);
    }

    @Override
    public double evaluate(MultiHeadRule rule, int numberOfLabelsInTheHead) {
        this.numberOfLabelsInTheHead = numberOfLabelsInTheHead;
        rawRuleValue = rule.getRawRuleValue();
        if (useAutomaticSettings)
            setParameters();
        boostedRuleValue = logValue();
        return boostedRuleValue;
    }

    private void setParameters() {
        b = getParameterBFromMaximum(maximum);
        a = getParameterA();
    }

    private double g(double x) {
        double t = Math.pow(b / Math.exp(1.0), 1.0/3.0);
        double logPart = - Math.log(b) + Math.log(Math.pow(x + t - 1.0, 3.0)) + 1.0;
        double xPart = (x + t - 1.0) * Math.log(b);
        return (logPart / xPart);
    }

    private double getParameterBFromMaximum(double x) {
        double numerator = x - 1;
        double oneThird = 1.0 / 3.0;
        double twoThirds = 2.0 / 3.0;
        double denominator = Math.exp(twoThirds) - (1 / Math.exp(oneThird));
        double value = numerator / denominator;
        return Math.pow(value, 3);
    }

    private double getParameterA() {
        double value = (boostAtMaximum - 1.0) / g(maximum);
        return value;
    }

    private double logValue() {
        double boost = logValue(numberOfLabelsInTheHead);
        double value = rawRuleValue * boost;
        return value;
    }

    private double logValue(double x) {
        double t = Math.pow(b / Math.exp(1), 1/3);
        double logPart = a * (- Math.log(b) + Math.log(Math.pow(x + t - 1, 3) + 1));
        double xPart = (x + t - 1) * Math.log(b);
        return (logPart / xPart) + 1;
    }

    public String toString() {
        return "LogXBoosting(" + a + "|" + b + ")";
    }

}
