package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * 1 - (FP + FN) / (P + N)
 */
public class HammingLoss extends ValueHeuristic {

    private static final long serialVersionUID = -4750847285013707969L;

    @Override
    public final double evaluateRule(final Rule rule) {
        return evaluateConfusionMatrix(rule.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        double distance = confusionMatrix.getNumberOfIncorrectClassified(); // FP + FN
        double numExamples = confusionMatrix.getNumberOfExamples(); // P + N

        if (numExamples == 0)
            return 0;
        return 1 - distance / numExamples;
    }

}