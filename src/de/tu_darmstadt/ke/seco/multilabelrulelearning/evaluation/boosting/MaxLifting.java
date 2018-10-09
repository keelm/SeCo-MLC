package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.LiftingStrategy;

public class MaxLifting extends LiftingStrategy {

    /**
     * Number of labels at which the lift function is maximal.
     */
    private double maximum;

    /**
     * Lift function value at the maximum. Therefore lift(maximum) = liftAtMaximum.
     */
    private double liftAtMaximum;

    /**
     * Curvature of the function while still abiding (1,1) and (maximum, liftAtMaximum).
     * A curvature of 1.0 corresponds to a linear function.
     */
    private double curvature;

    public MaxLifting(int maximumNumberOfLabels, double maximum, double liftAtMaximum, double curvature) {
        super(maximumNumberOfLabels);
        this.maximum = maximum;
        this.liftAtMaximum = liftAtMaximum;
        this.curvature = curvature;
        evaluateForAllHeadLengths();
    }

    @Override
    public double getMaximumLookaheadLift(int headSize, int lookahead) {
        if (maximum > headSize && maximum <= headSize + lookahead)
            return liftFunctionValues.get((int) maximum);
        return super.getMaximumLookaheadLift(headSize, lookahead);
    }

    @Override
    public double getMaximumLift(int headSize) {
        if (headSize <= maximum)
            return liftFunctionValues.get((int) (maximum > maximumNumberOfLabels ? maximumNumberOfLabels : maximum));
        return super.getMaximumLift(headSize);
    }

    @Override
    protected double lift(double x) {
        if (x <= maximum)
            return liftBeforeMaximum(x);
        return liftAfterMaximum(x);
    }

    private double liftBeforeMaximum(double x) {
        double exponent = 1.0 / curvature;
        double normalization = (x - 1.0) / (maximum - 1.0);
        double boost = 1.0 + Math.pow(normalization, exponent) * (liftAtMaximum - 1.0);
        return boost;
    }

    private double liftAfterMaximum(double x) {
        double exponent = 1.0 / curvature;
        double normalization = (x - maximumNumberOfLabels) / (maximumNumberOfLabels - maximum);
        double boost = 1.0 + Math.pow(-normalization, exponent) * (liftAtMaximum - 1.0);
        return boost;
    }

    public String toString() {
        return "Peak(" + maximum + "|" + liftAtMaximum + " | " + curvature + ")";
    }

}
