package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;

import java.util.Collection;

public class RuleDependentEvaluation extends EvaluationStrategy {

    @Override
    public Collection<Integer> getRelevantLabels(final int[] labelIndices, final Head head) {
        return head.getLabelIndices();
    }

}