/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * SingleHeadRuleSet.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Johannes Fürnkranz Modified by Viktor Seifert
 */

package de.tu_darmstadt.ke.seco.models;

import weka.core.Instance;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning.
 * <pruningDepth>
 * SingleHeadRuleSet implements the representation of a rule set.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 355 $
 */
public class SingleHeadRuleSet extends RuleSet<SingleHeadRule> {

    private static final long serialVersionUID = 1L;

    public double getDefaultPrediction() {
        return getDefaultRule().getPredictedValue();
    }

    public double classifyInstance(final Instance inst) throws Exception {
        for (int i = 0; i < numRules(); i++) {
            final double pred = getRule(i).classifyInstance(inst);
            if (!Utils.isMissingValue(pred))
                return pred;
        }

        return getDefaultPrediction();
    }

    @Override
    public ArrayList<SingleHeadRule> getCoveringRules(final Instance inst) {
        final ArrayList<SingleHeadRule> covRules = new ArrayList<SingleHeadRule>();

        for (int i = 0; i < numRules(); i++) {
            final SingleHeadRule r = getRule(i);
            final double pred = r.classifyInstance(inst);
            if (!Utils.isMissingValue(pred))
                covRules.add(r);
        }

        return covRules;
    }

    @Override
    public SingleHeadRule getFirstCoveringRule(final Instance inst) {
        for (int i = 0; i < numRules(); i++) {
            final SingleHeadRule r = getRule(i);
            final double pred = r.classifyInstance(inst);
            if (!Utils.isMissingValue(pred))
                return r;
        }

        return null;
    }

    @Override
    public String toString() {
        final StringBuilder stringBuilder = new StringBuilder();

        if (m_labelIndices != null) {
            SingleHeadRuleSet theory = this;
            int numRules = theory.size();
            int correctedNumRules = 0, numStopRules = 0;
            int numFullLabelRules = 0;
            int numPartialLabelRules = 0;
            int totalConditions = 0, totalLabelConditions = 0;
            int numPredictZero = 0;
            for (int i = 0; i < numRules; i++) {
                SingleHeadRule rule = theory.getRule(i);
                if ("magicSkipHead".equals(rule.getHead().getAttr().name())) {
                    numStopRules++;
                    continue; //dont count
                }
                if (rule.getHead().getValue() == 1.0)
                    numPredictZero++;
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
            }


            stringBuilder.append("number of rules......: " + correctedNumRules);
            stringBuilder.append("\nreferred attributes..: " + referredAttributes());
            stringBuilder.append("\naverage rule length..: " + (totalConditions) / (double) correctedNumRules);
            //		stringBuilder.append("\n#rulesWoStopRules....: " + correctedNumRules);
            stringBuilder.append("\n#stopRules...........: " + numStopRules);
            stringBuilder.append("\n#zeroLabelPredictions: " + numPredictZero);
            stringBuilder.append("\n#partLabelRulesNotFul: " + numPartialLabelRules);
            stringBuilder.append("\n#fullLabelRules......: " + numFullLabelRules);
            stringBuilder.append("\n#nonLabelRules.......: " + (correctedNumRules - numFullLabelRules - numPartialLabelRules));
            stringBuilder.append("\n%partLabelRulesNotFul: " + numPartialLabelRules / (double) correctedNumRules);
            stringBuilder.append("\n%fullLabelRules......: " + numFullLabelRules / (double) correctedNumRules);
            stringBuilder.append("\n%nonLabelRules.......: " + (correctedNumRules - numFullLabelRules - numPartialLabelRules) / (double) correctedNumRules);

            //		stringBuilder.append("\n#avgRuleLength(corr).: " + (totalConditions)/(double)correctedNumRules);
            stringBuilder.append("\nnumber of conditions.: " + numConditions());
            stringBuilder.append("\n#nonLabelConditions..: " + (totalConditions - totalLabelConditions));
            stringBuilder.append("\n#labelConditions.....: " + totalLabelConditions);
            stringBuilder.append("\n%nonLabelConditions..: " + (totalConditions - totalLabelConditions) / (double) totalConditions);
            stringBuilder.append("\n%labelConditions.....: " + totalLabelConditions / (double) totalConditions);


            //this is computed for the stacking approach
//			stats.createAvg("avgRulesPerRuleset","#rules");
//			stats.createAvg("avgConditionsPerRule", "#conditions");
//			stats.createAvg("ratioLabelConds","#labelConditions"); //in order to compute ratioLabelConditions
//			stats.createAvg("avgLabelCondsPerRule","#labelConditions2"); //in order to compute
//			stats.createAvg("ratioRulesWonlyLabelConditions","#rulesWonlyLabelConditions");
//			stats.createAvg("ratioRulesWOLabelConditions","#rulesWOLabelConditions"); //1-this = rules with partially label conditions
//			stats.createAvg("ratioRuleSetsWonlyLabelConditions","#ruleSetsWonlyLabelConditions");
//			stats.createAvg("ratioRuleSetsWOlabelConditions","#ruleSetsWOlabelConditions");
//			stats.createAvg("ratioModelsWonlyLabelConditions","#modelsWonlyLabelConditions");
//			stats.createAvg("ratioModelsWOlabelConditions","#modelsWOlabelConditions");
//			stats.set("avgLabelCondsPercentagePerRule",stats.getAvg("avgLabelCondsPercentagePerRule"));

        } else {
            stringBuilder.append("number of rules......: " + getRules().size());
            stringBuilder.append("\nnumber of conditions.: " + numConditions());
            stringBuilder.append("\nreferred attributes..: " + referredAttributes());
            stringBuilder.append("\naverage rule length..: " + averageLength());
        }


        if (getRules().size() > 0) {
            stringBuilder.append("\nRuleSet..............: ");
            boolean isFirst = true;
            for (final SingleHeadRule rule : getRules())
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

}
