package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import de.tu_darmstadt.ke.seco.utils.Logger;
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

            if (!covers || !areAllLabelsAlreadyPredicted(instance, head)) {
                for (int labelIndex : relevantLabels) {
                    aggregate(covers, head, instance, labelIndex, stats, null);
                }
            }

            if (covers) {
                coveredInstances.add(i);
            }
        }
        double h = heuristic.evaluateConfusionMatrix(stats);
        int numberOfLabels = head.size();
        // RELAXING CHANGE
        h = logValue(h, numberOfLabels);

        rule.setRuleValue(heuristic, h);
        return new MicroAveragingMetaData(coveredInstances, stats);
    }

    private int log = 25;

    private double logValue(double heuristic, int numberOfLabels) {
        heuristic *= (Math.log(numberOfLabels + log - 1) / Math.log(log));
        return heuristic;
    }

    private int times = 10;

    private double logTimesValue(double heuristic, int numberOfLabels) {
        heuristic *= (Math.log(10 * (numberOfLabels + log - 1)) / Math.log(log));
        return heuristic;
    }

    private double quadrValue(double heuristic, int numberOfLabels) {
        heuristic *= 1 + Math.pow((1 / ((numberOfLabels+1) / 2)), 2);
        return heuristic;
    }

    private double a = 0.1;

    private double tradeOffValue(double heuristic, int numberOfLabels) {
        heuristic = (1 - a) * heuristic+ a * numberOfLabels;
        return heuristic;
    }

    @Override
    public final String toString() {
        return "mm";
    }

}