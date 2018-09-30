package de.tu_darmstadt.ke.seco.multilabelrulelearning.prepending.evaluation.averaging;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instance;

import java.util.Collection;
import java.util.LinkedList;

public class MicroAveraging extends AveragingStrategy {

    private static class MicroAveragingMetaData extends MetaData {

        MicroAveragingMetaData(final Collection<Integer> coveredInstances,
                               final TwoClassConfusionMatrix uncoveredStats) {
            super(coveredInstances, uncoveredStats);
        }

    }

    @Override
    protected final MetaData evaluate(final Instances instances, final MultiHeadRule rule, final Heuristic heuristic,
                                      final Collection<Integer> relevantLabels, final MetaData metaData,
                                      final TwoClassConfusionMatrix stats) {
        Collection<Integer> coveredInstances = new LinkedList<>();
        Head head = rule.getHead();
        boolean refinement = metaData instanceof MicroAveragingMetaData;

        for (int i : refinement ? metaData.coveredInstances : instancesIterable(instances)) {
            Instance instance = instances.get(i);
            boolean covers = rule.covers(instance);

            // TODO: condition must be omitted for prepending!
            //if (!covers || !areAllLabelsAlreadyPredicted(instance, head)) {
                for (int labelIndex : relevantLabels) {
                    aggregatePrepending(covers, head, instance, labelIndex, stats, null);
                }
            //}

            if (covers) {
                coveredInstances.add(i);
            }
        }
        double h = heuristic.evaluateConfusionMatrix(stats);

        rule.setRuleValue(heuristic, h);
        rule.setRawRuleValue(h);
        return new MicroAveragingMetaData(coveredInstances, stats);
    }

    @Override
    public final String toString() {
        return "mm";
    }

}