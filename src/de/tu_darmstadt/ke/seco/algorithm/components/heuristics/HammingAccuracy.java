package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.RuleDependentEvaluation;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * (TP + TN) / (P + N)
 */
public class HammingAccuracy extends ValueHeuristic {

    private static final long serialVersionUID = -4750847285013707969L;

    @Override
    public final double evaluateRule(final Rule rule) {
        return evaluateConfusionMatrix(rule.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        double correctly = confusionMatrix.getNumberOfCorrectlyClassified();
        double numExamples = confusionMatrix.getNumberOfExamples(); // P + N

        if (numExamples == 0)
            return 0;
        return correctly / numExamples;
    }

    @Override
    public final Characteristic getCharacteristic(final EvaluationStrategy evaluationStrategy,
                                                  final AveragingStrategy averagingStrategy) {
        if (evaluationStrategy instanceof RuleDependentEvaluation) {
            return Characteristic.DECOMPOSABLE;
        } else {
            return Characteristic.ANTI_MONOTONOUS;
        }
    }

}