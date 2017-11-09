package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic.Characteristic;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

import java.util.Collection;
import java.util.LinkedHashSet;

public class MultiLabelEvaluation {

    public static abstract class MetaData {

        public final Collection<Integer> coveredInstances;

        public final TwoClassConfusionMatrix stats;

        public MetaData(final Collection<Integer> coveredInstances, final TwoClassConfusionMatrix uncoveredStats) {
            this.coveredInstances = coveredInstances;
            this.stats = uncoveredStats;
        }

    }

    private final Heuristic heuristic;

    private final EvaluationStrategy evaluationStrategy;

    private final AveragingStrategy averagingStrategy;

    public MultiLabelEvaluation(final Heuristic heuristic, final EvaluationStrategy evaluationStrategy, final
    AveragingStrategy averagingStrategy) {
        this.heuristic = heuristic;
        this.evaluationStrategy = evaluationStrategy;
        this.averagingStrategy = averagingStrategy;
    }

    public final MetaData evaluate(final Instances instances, final LinkedHashSet<Integer> labelIndices, final MultiHeadRule rule,
                                   final MetaData metaData) {
        Collection<Integer> relevantLabels = evaluationStrategy.getRelevantLabels(labelIndices, rule.getHead());
        return averagingStrategy.evaluate(instances, rule, heuristic, relevantLabels, metaData);
    }

    public final Heuristic getHeuristic() {
        return heuristic;
    }

    public final EvaluationStrategy getEvaluationStrategy() {
        return evaluationStrategy;
    }

    public final AveragingStrategy getAveragingStrategy() {
        return averagingStrategy;
    }

    public final Characteristic getCharacteristic() {
        return getHeuristic().getCharacteristic(getEvaluationStrategy(), getAveragingStrategy());
    }

}