package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

@Deprecated
public class LogAlphaBoosting extends BoostingStrategy {

    private double m = 10; // gradient
    private double a = 1.16;

    private int log = 5;

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
        //double logValue = (m * (Math.log(log - 1 + x)) / Math.log((log / a) * Math.pow(a, x))) - (m - 1);
        double zahler = Math.log(log - 1 + x);
        double nenner = Math.log((log / a) * Math.pow(a, x));
        double logValue = m * (zahler / nenner) - (m - 1);
        return logValue;
    }

    public String toString() {
        return "LogAlphaBoosting(" + m + "|" + log + "|" + a + ")";
    }

}
