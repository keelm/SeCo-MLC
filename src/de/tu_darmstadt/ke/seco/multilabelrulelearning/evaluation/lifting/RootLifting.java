package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.lifting;

import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.LiftingStrategy;

public class RootLifting extends LiftingStrategy {

    /** Parameter of the lift function. */
    private double k;


    public RootLifting(int maximumNumberOfLabels, double k) {
        super(maximumNumberOfLabels);
        this.k = k;
        evaluateForAllHeadLengths();
    }

    public RootLifting(int maximumNumberOfLabels, double point, double boostAtPoint) {
        super(maximumNumberOfLabels);
        this.k = Math.log(point) / Math.log(boostAtPoint);
        evaluateForAllHeadLengths();
    }


    @Override
    protected double lift(double x) {
        return Math.pow(x, 1.0 / k);
    }


    public String toString() {
        return "Root(" + k + ")";
    }


}
