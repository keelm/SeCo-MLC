package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.boosting;

import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.BoostingStrategy;

public class MaxBoosting extends BoostingStrategy {

    /**
     * Number of labels at which the boost function is maximal.
     */
    private double maximum;

    /**
     * Boost function value at the maximum. Therefore boost(maximum) = boostAtMaximum.
     */
    private double boostAtMaximum;

    /**
     * Curvature of the function while still abiding (1,1) and (maximum, boostAtMaximum).
     * A curvature of 1.0 corresponds to a linear function.
     */
    private double curvature;

    public MaxBoosting(int maximumNumberOfLabels, double maximum, double boostAtMaximum, double curvature) {
        super(maximumNumberOfLabels);
        this.maximum = maximum;
        this.boostAtMaximum = boostAtMaximum;
        this.curvature = curvature;
        evaluateForAllHeadLengths();
    }

    @Override
    public double getMaximumLookaheadBoost(int headSize, int lookahead) {
        if (maximum >= headSize && maximum <= headSize + lookahead)
            return boostFunctionValues.get(maximum);
        return super.getMaximumLookaheadBoost(headSize, lookahead);
    }

    @Override
    public double getMaximumBoost(int headSize) {
        if (headSize <= maximum)
            return boostFunctionValues.get(maximum);
        return super.getMaximumBoost(headSize);
    }

    @Override
    protected double boost(double x) {
        if (x <= maximum)
            return boostBeforeMaximum(x);
        return boostAfterMaximum(x);
    }

    private double boostBeforeMaximum(double x) {
        double exponent = 1.0 / curvature;
        double normalization = (x - 1.0) / (maximum - 1.0);
        double boost = 1.0 + Math.pow(normalization, exponent) * (boostAtMaximum - 1.0);
        return boost;
    }

    private double boostAfterMaximum(double x) {
        double exponent = 1.0 / curvature;
        double normalization = (x - maximumNumberOfLabels) / (maximumNumberOfLabels - maximum);
        double boost = 1.0 + Math.pow(-normalization, exponent) * (boostAtMaximum - 1.0);
        return boost;
    }

    public String toString() {
        return "MaxBoosting(" + maximum + "|" + boostAtMaximum + " | " + curvature + ")";
    }

}
