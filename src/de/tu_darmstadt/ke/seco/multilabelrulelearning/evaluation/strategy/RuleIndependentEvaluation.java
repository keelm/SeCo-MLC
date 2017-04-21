package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;

import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RuleIndependentEvaluation extends EvaluationStrategy {

    @Override
    public final Collection<Integer> getRelevantLabels(final int[] labelIndices, final Head head) {
        return IntStream.of(labelIndices).boxed().collect(Collectors.toList());
    }

}