package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class LinLogBoosting extends BoostingStrategy {

    private double u = 30; // switching point
    private double m = 0.05; // gradient

    private int log = 8;

    private double numberOfLabelsInTheHead;

    private double rawRuleValue;
    private double boostedRuleValue;

    private double linSwitchingPointValue = 0;
    private double logSwitchingPointValue = 0;

    @Override
    public void evaluate(MultiHeadRule rule) {
        numberOfLabelsInTheHead = rule.getHead().size();
        rawRuleValue = rule.getRawRuleValue();
        boostedRuleValue = linLogValue();
        rule.setBoostedRuleValue(boostedRuleValue);
    }

    @Override
    public double evaluate(MultiHeadRule rule, double numberOfLabelsInTheHead) {
        this.numberOfLabelsInTheHead = numberOfLabelsInTheHead;
        rawRuleValue = rule.getRawRuleValue();
        boostedRuleValue = linLogValue();
        return boostedRuleValue;
    }

    private double linLogValue() {
        double boost = numberOfLabelsInTheHead < u ? linBoost(numberOfLabelsInTheHead) : logBoost(numberOfLabelsInTheHead);
        double value = rawRuleValue * (1 + boost);
        return value;
    }

    private double linBoost(double x) {
        return m * x;
    }

    private double logBoost(double x) {
        linSwitchingPointValue = linSwitchingPointValue == 0 ? linBoost(u) : linSwitchingPointValue;
        logSwitchingPointValue = logSwitchingPointValue == 0 ? logValue(u) : logSwitchingPointValue;
        double boost = linSwitchingPointValue - logSwitchingPointValue + logValue(x);
        return boost;
    }

    private double logValue(double x) {
        double logValue = (Math.log(x + log - 1) / Math.log(log));
        logValue = logValue < 1 ? 1 : logValue;
        return logValue;
    }

}
