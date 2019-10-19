package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instance;

import java.util.Collection;
import java.util.LinkedList;

public class ExampleBasedAveraging extends AveragingStrategy {

    private class ExampleBasedAveragingMetaData extends MetaData {
        private final double uncoveredH;

        ExampleBasedAveragingMetaData(final Collection<Integer> coveredInstances,
                                      final TwoClassConfusionMatrix uncoveredStats,
                                      final double uncoveredH) {
            super(coveredInstances, uncoveredStats);
            this.uncoveredH = uncoveredH;
        }

    }

    @Override
    protected final MetaData evaluate(final Instances instances, final MultiHeadRule rule, final Heuristic heuristic,
                                      final Collection<Integer> relevantLabels, final MetaData metaData,
                                      final TwoClassConfusionMatrix stats, final TwoClassConfusionMatrix recall) {
        Collection<Integer> coveredInstances = new LinkedList<>();
        Head head = rule.getHead();
        double uncoveredH = 0;
        double h = 0;
        boolean refinement = metaData instanceof ExampleBasedAveragingMetaData;

        if (refinement) {
            ExampleBasedAveragingMetaData exampleBasedAveragingMetaData = (ExampleBasedAveragingMetaData) metaData;
            uncoveredH = exampleBasedAveragingMetaData.uncoveredH;
            h += uncoveredH;
        }

        for (int i : refinement ? metaData.coveredInstances : instancesIterable(instances)) {
            Instance instance = instances.get(i);
            boolean covers = rule.covers(instance);

            if (!covers || !areAllLabelsAlreadyPredicted(instance, head)) {
                TwoClassConfusionMatrix confusionMatrix = new TwoClassConfusionMatrix();

                for (int labelIndex : relevantLabels) {
                    aggregate(covers, head, instance, labelIndex, confusionMatrix, stats, recall);
                }

                double exampleWiseH = heuristic.evaluateConfusionMatrix(confusionMatrix);

                if (!covers) {
                    uncoveredH += exampleWiseH;
                }

                h += exampleWiseH;
            }

            if (covers) {
                coveredInstances.add(i);
            }
        }

        h = h / (double) instances.size();
        rule.setRuleValue(heuristic, h);
        return new ExampleBasedAveragingMetaData(coveredInstances, stats, uncoveredH);
    }

    @Override
    public final String toString() {
        return "Mm";
    }

}