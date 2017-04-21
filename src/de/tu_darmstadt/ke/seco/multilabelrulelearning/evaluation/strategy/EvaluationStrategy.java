package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;

import java.util.Collection;

public abstract class EvaluationStrategy {

    public static final String RULE_DEPENDENT = "rule-dependent";

    public static final String RULE_INDEPENDENT = "rule-independent";

    public abstract Collection<Integer> getRelevantLabels(int[] labelIndices, Head head);

    public static EvaluationStrategy create(final String strategy) {
        if (strategy.equalsIgnoreCase(RULE_DEPENDENT)) {
            return new RuleDependentEvaluation();
        } else if (strategy.equalsIgnoreCase(RULE_INDEPENDENT)) {
            return new RuleIndependentEvaluation();
        }

        throw new IllegalArgumentException("Invalid evaluation strategy: " + strategy);
    }

}