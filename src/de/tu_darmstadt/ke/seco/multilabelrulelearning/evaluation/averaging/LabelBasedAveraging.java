package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instance;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

public class LabelBasedAveraging extends AveragingStrategy {

    private class LabelBasedAveragingMetaData extends MetaData {

        private Map<Integer, TwoClassConfusionMatrix> uncoveredLabelWiseStats;

        public LabelBasedAveragingMetaData(final Collection<Integer> coveredInstances,
                                           final TwoClassConfusionMatrix uncoveredStats,
                                           final Map<Integer, TwoClassConfusionMatrix> uncoveredLabelWiseStats) {
            super(coveredInstances, uncoveredStats);
            this.uncoveredLabelWiseStats = uncoveredLabelWiseStats;
        }

    }

    @Override
    protected final MetaData evaluate(final Instances instances, final MultiHeadRule rule, final Heuristic heuristic,
                                      final Collection<Integer> relevantLabels, final MetaData metaData,
                                      final TwoClassConfusionMatrix stats) {
        Collection<Integer> coveredInstances = new LinkedList<>();
        Map<Integer, TwoClassConfusionMatrix> uncoveredLabelWiseStats = new HashMap<>();
        double h = 0;
        Head head = rule.getHead();
        boolean refinement = metaData instanceof LabelBasedAveragingMetaData;
        boolean firstLabel = true;

        if (refinement) {
            LabelBasedAveragingMetaData labelBasedAveragingMetaData = (LabelBasedAveragingMetaData) metaData;
            uncoveredLabelWiseStats = labelBasedAveragingMetaData.uncoveredLabelWiseStats;
        }

        for (int labelIndex : relevantLabels) {
            TwoClassConfusionMatrix confusionMatrix = new TwoClassConfusionMatrix();
            TwoClassConfusionMatrix labelWiseStats = uncoveredLabelWiseStats.get(labelIndex);

            if (labelWiseStats != null) {
                confusionMatrix.addTrueNegatives(labelWiseStats.getNumberOfTrueNegatives());
                confusionMatrix.addFalseNegatives(labelWiseStats.getNumberOfFalseNegatives());
            }

            for (int i : refinement ? metaData.coveredInstances : instancesIterable(instances)) {
                Instance instance = instances.get(i);
                boolean covers = rule.covers(instance);

                if (!covers || !areAllLabelsAlreadyPredicted(instance, head)) {
                    aggregate(covers, head, instance, labelIndex, confusionMatrix, stats);
                }

                if (firstLabel && covers) {
                    coveredInstances.add(i);
                }
            }

            h += heuristic.evaluateConfusionMatrix(confusionMatrix);
            uncoveredLabelWiseStats.put(labelIndex, confusionMatrix);
            firstLabel = false;
        }

        h = h / (double) relevantLabels.size();
        rule.setRuleValue(heuristic, h);
        return new LabelBasedAveragingMetaData(coveredInstances, stats, uncoveredLabelWiseStats);
    }

    @Override
    public final String toString() {
        return "mM";
    }

}