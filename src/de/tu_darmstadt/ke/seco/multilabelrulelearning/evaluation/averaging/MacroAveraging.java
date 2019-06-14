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

public class MacroAveraging extends AveragingStrategy {

    private class MacroAveragingMetaData extends MetaData {

        private double uncoveredH;

        public MacroAveragingMetaData(final Collection<Integer> coveredInstances,
                                      final TwoClassConfusionMatrix uncoveredStats,
                                      final double uncoveredH) {
            super(coveredInstances, uncoveredStats);
            this.uncoveredH = uncoveredH;
        }

    }

    @Override
    protected final MetaData evaluate(final Instances instances, final MultiHeadRule rule, final Heuristic heuristic,
                                      final Collection<Integer> relevantLabels, final MetaData metaData,
                                      final TwoClassConfusionMatrix stats) {

        Collection<Integer> coveredInstances = new LinkedList<>();
        double h = 0;
        double uncoveredH = 0;
        Head head = rule.getHead();
        boolean refinement = metaData instanceof MacroAveragingMetaData;

        if (refinement) {
            MacroAveragingMetaData macroAveragingMetaData = (MacroAveragingMetaData) metaData;
            uncoveredH = macroAveragingMetaData.uncoveredH;
            h += uncoveredH;
        }

        for (int i : refinement ? metaData.coveredInstances : instancesIterable(instances)) {
            Instance instance = instances.get(i);
            boolean covers = rule.covers(instance);

            if (!covers || !areAllLabelsAlreadyPredicted(instance, head)) {
                double exampleWiseH = 0;
                for (int labelIndex : relevantLabels) {
                    TwoClassConfusionMatrix confusionMatrix = new TwoClassConfusionMatrix();
                    aggregate(covers, head, instance, labelIndex, confusionMatrix, stats);
                    exampleWiseH += heuristic.evaluateConfusionMatrix(confusionMatrix);
                }

                exampleWiseH = exampleWiseH / (double) relevantLabels.size();

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
        rule.setRawRuleValue(h);
        return new MacroAveragingMetaData(coveredInstances, stats, uncoveredH);
    }

    @Override
    public final String toString() {
        return "MM";
    }

}