package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class LLNBoosting extends BoostingStrategy {

    /**
     * Parameter of the boost function.
     */
    private double l;

    public LLNBoosting(int maximumNumberOfLabels, double l) {
        super(maximumNumberOfLabels);
        this.l = l;
        evaluateForAllHeadLengths();
    }

    @Override
    protected double boost(double x) {
        double logValue = 1.0 + l * Math.log(x);
        return logValue;
    }

    public String toString() {
        return "LLNBoosting(" + l + "|)";
    }

}
