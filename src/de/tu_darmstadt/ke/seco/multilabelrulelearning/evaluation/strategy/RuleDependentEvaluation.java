package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;

public class RuleDependentEvaluation extends EvaluationStrategy {

    @Override
    public Collection<Integer> getRelevantLabels(final LinkedHashSet<Integer> labelIndices, final Head head) {
        return head.getLabelIndices();
    }

}