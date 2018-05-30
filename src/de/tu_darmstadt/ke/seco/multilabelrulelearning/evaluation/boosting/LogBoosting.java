package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class LogBoosting extends BoostingStrategy {

    private double m = 1; // gradient

    private int log = 15;

    private double numberOfLabelsInTheHead;

    private double rawRuleValue;
    private double boostedRuleValue;

    @Override
    public void evaluate(MultiHeadRule rule) {
        numberOfLabelsInTheHead = rule.getHead().size();
        rawRuleValue = rule.getRawRuleValue();
        boostedRuleValue = logValue();
        rule.setBoostedRuleValue(boostedRuleValue);
    }

    @Override
    public double evaluate(MultiHeadRule rule, double numberOfLabelsInTheHead) {
        this.numberOfLabelsInTheHead = numberOfLabelsInTheHead;
        rawRuleValue = rule.getRawRuleValue();
        boostedRuleValue = logValue();
        return boostedRuleValue;
    }

    private double logValue() {
        double boost = logValue(numberOfLabelsInTheHead);
        double value = rawRuleValue * boost;
        return value;
    }

    private double logValue(double x) {
        double logValue = (m * (Math.log(log - 1 + x)) / Math.log(log)) - (m - 1);
        // double logValue = Math.log(log - 1 + m * x) / Math.log(log);
        return logValue;
    }

    public String toString() {
        return "LogBoosting(" + m + "|" + log + ")";
    }

}
