package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class RootBoosting extends BoostingStrategy {

    /**
     * Parameter of the boost function.
     */
    private double k;

    public RootBoosting(int maximumNumberOfLabels, double k) {
        super(maximumNumberOfLabels);
        this.k = k;
        evaluateForAllHeadLengths();
    }

    public RootBoosting(int maximumNumberOfLabels, double point, double boostAtPoint) {
        super(maximumNumberOfLabels);
        this.k = Math.log(point) / Math.log(boostAtPoint);
        evaluateForAllHeadLengths();
    }

    @Override
    protected double boost(double x) {
        return Math.pow(x, 1.0 / k);
    }

    public String toString() {
        return "RootBoosting(" + k + ")";
    }


}
