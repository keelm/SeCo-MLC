package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.lifting;

import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.LiftingStrategy;

public class KLNLifting extends LiftingStrategy {

    /** Parameter of the lift function. */
    private double l;


    public KLNLifting(int maximumNumberOfLabels, double l) {
        super(maximumNumberOfLabels);
        this.l = l;
        evaluateForAllHeadLengths();
    }

    public KLNLifting(int maximumNumberOfLabels, double point, double boostAtPoint) {
        super(maximumNumberOfLabels);
        this.l = (boostAtPoint - 1.0) / Math.log(point);
        evaluateForAllHeadLengths();
    }


    @Override
    protected double lift(double x) {
        double logValue = 1.0 + l * Math.log(x);
        return logValue;
    }


    public String toString() {
        return "KLN(" + l + "|)";
    }

}
