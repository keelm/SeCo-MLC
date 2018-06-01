package de.tu_darmstadt.ke.seco.models;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.MulticlassCovering;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Arrays;

import static de.tu_darmstadt.ke.seco.Main.csvWriter;

public class MultiHeadRuleSet extends RuleSet<MultiHeadRule> {

    private static final long serialVersionUID = 1L;

    public double[] getDefaultPrediction() {
        return getDefaultRule().getPredictedValue();
    }

    public double[] classifyInstance(final Instance inst) throws Exception {
        for (int i = 0; i < numRules(); i++) {
            final double[] pred = getRule(i).classifyInstance(inst);
            if (pred != null)
                return pred;
        }
        
        return getDefaultPrediction();
    }

    @Override
    public ArrayList<MultiHeadRule> getCoveringRules(final Instance inst) {
        final ArrayList<MultiHeadRule> covRules = new ArrayList<>();

        for (int i = 0; i < numRules(); i++) {
            final MultiHeadRule r = getRule(i);
            final double[] pred = r.classifyInstance(inst);
            if (pred != null)
                covRules.add(r);
        }

        return covRules;
    }

    @Override
    public MultiHeadRule getFirstCoveringRule(final Instance inst) {
        for (int i = 0; i < numRules(); i++) {
            final MultiHeadRule r = getRule(i);
            final double[] pred = r.classifyInstance(inst);
            if (pred != null)
                return r;
        }

        return null;
    }

    @Override
    public String toString() {
        final StringBuilder stringBuilder = new StringBuilder();

        if (m_labelIndices != null) {
            MultiHeadRuleSet theory = this;
            int numRules = theory.size();
            int correctedNumRules = 0, numStopRules = 0;
            int numFullLabelRules = 0;
            int numPartialLabelRules = 0;
            int totalConditions = 0, totalLabelConditions = 0;
            int numPredictZero = 0;
            int numMultiHeadRules = 0;
            int totalLabels = 0;
            int totalLabelsMultiHeadRules = 0;
            for (int i = 0; i < numRules; i++) {
                MultiHeadRule rule = theory.getRule(i);
                Head head = rule.getHead();

                if (head.size() > 0 && "magicSkipHead".equals(head.iterator().next().getAttr().name())) {
                    numStopRules++;
                    continue; //dont count
                }

                for (Condition cond : head) {
                    if (cond.getValue() == 1.0)
                        numPredictZero++;
                }
                if (head.size() > 1) {
                    totalLabelsMultiHeadRules += head.size();
                    numMultiHeadRules++;
                }
                correctedNumRules++;
                int numConditions = rule.length();
                int numLabelConditions = 0;
                for (int j = 0; j < numConditions; j++) {
                    Condition condition = rule.getCondition(j);
                    int attIndex = condition.getAttr().index();
                    if (Arrays.binarySearch(m_labelIndices, attIndex) >= 0)
                        numLabelConditions++;
                }
                if (numLabelConditions == numConditions && numConditions > 0)
                    numFullLabelRules++;
                else if (numLabelConditions > 0)
                    numPartialLabelRules++;
                totalConditions += numConditions;
                totalLabelConditions += numLabelConditions;
                totalLabels += head.size();
            }

            stringBuilder.append("number of rules....................: " + correctedNumRules);
            stringBuilder.append("\nreferred attributes..............: " + referredAttributes());
            stringBuilder.append("\naverage rule length..............: " + (totalConditions) / (double) correctedNumRules);
            stringBuilder.append("\n#stopRules.......................: " + numStopRules);
            stringBuilder.append("\n#zeroLabelPredictions............: " + numPredictZero);
            stringBuilder.append("\n#partLabelRulesNotFul............: " + numPartialLabelRules);
            stringBuilder.append("\n#fullLabelRules..................: " + numFullLabelRules);
            stringBuilder.append("\n#nonLabelRules...................: " +
                    (correctedNumRules - numFullLabelRules - numPartialLabelRules));
            stringBuilder.append("\n%partLabelRulesNotFul............: " + numPartialLabelRules / (double) correctedNumRules);
            stringBuilder.append("\n%fullLabelRules..................: " + numFullLabelRules / (double) correctedNumRules);
            stringBuilder.append("\n%nonLabelRules...................: " +
                    (correctedNumRules - numFullLabelRules - numPartialLabelRules) / (double) correctedNumRules);
            stringBuilder.append("\nnumber of conditions.............: " + numConditions());
            stringBuilder.append("\n#nonLabelConditions..............: " + (totalConditions - totalLabelConditions));
            stringBuilder.append("\n#labelConditions.................: " + totalLabelConditions);
            stringBuilder.append("\n%nonLabelConditions..............: " +
                    (totalConditions - totalLabelConditions) / (double) totalConditions);
            stringBuilder.append("\n%labelConditions.................: " + totalLabelConditions / (double) totalConditions);
            stringBuilder.append("\n#multiHeadRules..................: " + numMultiHeadRules);
            stringBuilder.append("\n%multiHeadRules..................: " + numMultiHeadRules / (double) correctedNumRules);
            stringBuilder.append("\naverage #labels..................: " + totalLabels / (double) correctedNumRules);
            stringBuilder.append("\naverage #labels per multiHeadRule: " + (numMultiHeadRules == 0 ? 0 : totalLabelsMultiHeadRules / (double) numMultiHeadRules));
            if (MulticlassCovering.finished && !csvWritten) {
                csvWritten = true;
                csvWriter.writeNext(new String[]{"number of rules", Integer.toString(correctedNumRules)});
                csvWriter.writeNext(new String[]{"referred attributes", Integer.toString(referredAttributes())});
                csvWriter.writeNext(new String[]{"average rule length", Double.toString((totalConditions) / (double) correctedNumRules)});
                csvWriter.writeNext(new String[]{"#stopRules", Integer.toString(numStopRules)});
                csvWriter.writeNext(new String[]{"#zeroLabelPredictions", Integer.toString(numPredictZero)});
                csvWriter.writeNext(new String[]{"#partLabelRulesNotFul", Integer.toString(numPartialLabelRules)});
                csvWriter.writeNext(new String[]{"#fullLabelRules", Integer.toString(numFullLabelRules)});
                csvWriter.writeNext(new String[]{"#nonLabelRules",
                        Integer.toString((correctedNumRules - numFullLabelRules - numPartialLabelRules))});
                csvWriter.writeNext(new String[]{"%partLabelRulesNotFul",  Double.toString(numPartialLabelRules / (double) correctedNumRules)});
                csvWriter.writeNext(new String[]{"%fullLabelRules" ,  Double.toString(numFullLabelRules / (double) correctedNumRules)});
                csvWriter.writeNext(new String[]{"%nonLabelRules",
                        Double.toString((correctedNumRules - numFullLabelRules - numPartialLabelRules) / (double) correctedNumRules)});
                csvWriter.writeNext(new String[]{"number of conditions",   Integer.toString(numConditions())});
                csvWriter.writeNext(new String[]{"#nonLabelConditions",   Integer.toString((totalConditions - totalLabelConditions))});
                csvWriter.writeNext(new String[]{"#labelConditions",  Double.toString(totalLabelConditions)});
                csvWriter.writeNext(new String[]{"%nonLabelConditions",
                        Double.toString((totalConditions - totalLabelConditions) / (double) totalConditions)});
                csvWriter.writeNext(new String[]{"%labelConditions",  Double.toString(totalLabelConditions / (double) totalConditions)});
                csvWriter.writeNext(new String[]{"#multiHeadRules",   Integer.toString(numMultiHeadRules)});
                csvWriter.writeNext(new String[]{"%multiHeadRules",  Double.toString(numMultiHeadRules / (double) correctedNumRules)});
                csvWriter.writeNext(new String[]{"average #labels",  Double.toString(totalLabels / (double) correctedNumRules)});
                csvWriter.writeNext(new String[]{"average #labels per multiHeadRule",  Double.toString((numMultiHeadRules == 0 ? 0 : totalLabelsMultiHeadRules / (double) numMultiHeadRules))});

            }

        } else {
            stringBuilder.append("number of rules....................: " + getRules().size());
            stringBuilder.append("\nnumber of conditions.............: " + numConditions());
            stringBuilder.append("\nreferred attributes..............: " + referredAttributes());
            stringBuilder.append("\naverage rule length..............: " + averageLength());
        }

        if (getRules().size() > 0) {
            stringBuilder.append("\nRuleSet..............: ");
            boolean isFirst = true;
            for (final MultiHeadRule rule : getRules())
                if (isFirst) {
                    stringBuilder.append(rule);
                    isFirst = false;
                } else {
                    stringBuilder.append("\n                       " + rule);
                }
        }

        stringBuilder.append("\ndefaultRule..........: " + getDefaultRule());
        return stringBuilder.toString();
    }

    private static boolean csvWritten = false;

}
