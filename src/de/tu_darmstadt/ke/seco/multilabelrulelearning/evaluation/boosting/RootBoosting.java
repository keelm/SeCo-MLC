package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class RootBoosting extends BoostingStrategy {

    private double k = 4.1;

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
        return Math.pow(x, 1.0 / k);
    }

    public String toString() {
        return "RootBoosting(" + k + ")";
    }


}
