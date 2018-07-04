package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging;

import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Condition;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.DenseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.SparseInstanceWrapper;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.MultiLabelEvaluation.MetaData;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instance;
import weka.core.Utils;

import java.util.Collection;
import java.util.Iterator;

public abstract class AveragingStrategy {

    public static final String MICRO_AVERAGING = "micro-averaging";

    public static final String LABEL_BASED_AVERAGING = "label-based-averaging";

    public static final String EXAMPLE_BASED_AVERAGING = "example-based-averaging";

    public static final String MACRO_AVERAGING = "macro-averaging";

    private double getLabelValue(final Instance instance, final int labelIndex) {
        Instance wrappedInstance =
                instance instanceof DenseInstanceWrapper ? ((DenseInstanceWrapper) instance).getWrappedInstance() :
                        ((SparseInstanceWrapper) instance).getWrappedInstance();
        return wrappedInstance.value(labelIndex);
    }

    final Iterable<Integer> instancesIterable(final Instances instances) {
        return () -> new Iterator<Integer>() {

            private int i = 0;

            @Override
            public boolean hasNext() {
                return i < instances.numInstances();
            }

            @Override
            public Integer next() {
                return i++;
            }

        };
    }

    final boolean areAllLabelsAlreadyPredicted(final Instance instance, final Head head) {
        for (Condition labelAttribute : head) {
            if (Utils.isMissingValue(instance.value(labelAttribute.getAttr().index()))) {
                return false;
            }
        }

        return true;
    }

    final void aggregate(final boolean covers, final Head head, final Instance instance, final int labelIndex,
                         final TwoClassConfusionMatrix confusionMatrix, final TwoClassConfusionMatrix stats) {
        double labelValue = getLabelValue(instance, labelIndex);

        if (covers) {
            Condition labelAttribute = head.getCondition(labelIndex);

            if (labelAttribute != null) {
                if (labelAttribute.getValue() != labelValue) {
                    confusionMatrix.addFalsePositives(instance.weight());

                    if (stats != null) {
                        stats.addFalsePositives(instance.weight());
                    }
                } else {
                    confusionMatrix.addTruePositives(instance.weight());

                    if (stats != null) {
                        stats.addTruePositives(instance.weight());
                    }
                }
            } else {
                if (labelValue == 1.0) {
                    confusionMatrix.addFalsePositives(instance.weight());

                    if (stats != null) {
                        stats.addFalsePositives(instance.weight());
                    }
                } else {
                    confusionMatrix.addTruePositives(instance.weight());

                    if (stats != null) {
                        stats.addTruePositives(instance.weight());
                    }
                }
            }
        } else {
            //Condition labelAttribute = head.getCondition(labelIndex);
            if (labelValue == 1.0) {
                confusionMatrix.addFalseNegatives(instance.weight());

                if (stats != null) {
                    stats.addFalseNegatives(instance.weight());
                }
            } else {
                confusionMatrix.addTrueNegatives(instance.weight());

                if (stats != null) {
                    stats.addTrueNegatives(instance.weight());
                }
            }
        }
    }

    public final MetaData evaluate(final Instances instances, final MultiHeadRule rule,
                                   final Heuristic heuristic,
                                   final Collection<Integer> relevantLabels, final MetaData metaData) {
        TwoClassConfusionMatrix stats = new TwoClassConfusionMatrix();

        if (metaData != null) {
            stats.addTrueNegatives(metaData.stats.getNumberOfTrueNegatives());
            stats.addFalseNegatives(metaData.stats.getNumberOfFalseNegatives());
        }

        MetaData result = evaluate(instances, rule, heuristic, relevantLabels, metaData, stats);
        rule.setStats(stats);
        return result;
    }

    protected abstract MetaData evaluate(final Instances instances, final MultiHeadRule rule, final Heuristic heuristic,
                                         final Collection<Integer> relevantLabels, final MetaData metaData,
                                         final TwoClassConfusionMatrix stats);

    public static AveragingStrategy create(final String strategy) {
        if (strategy.equalsIgnoreCase(MICRO_AVERAGING)) {
            return new MicroAveraging();
        } else if (strategy.equalsIgnoreCase(LABEL_BASED_AVERAGING)) {
            return new LabelBasedAveraging();
        } else if (strategy.equalsIgnoreCase(EXAMPLE_BASED_AVERAGING)) {
            return new ExampleBasedAveraging();
        } else if (strategy.equalsIgnoreCase(MACRO_AVERAGING)) {
            return new MacroAveraging();
        }

        throw new IllegalArgumentException("Invalid averaging strategy: " + strategy);
    }

}