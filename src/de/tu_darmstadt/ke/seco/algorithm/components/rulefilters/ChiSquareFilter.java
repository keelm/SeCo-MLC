/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * ChiSquareFilter.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 03.11.2004, 10:35
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulefilters;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions.LikelihoodRatio;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.utils.Logger;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Iterator;
import java.util.TreeSet;

/**
 * This filter will check, whether a rule is significant in comparison to its predecessor.
 *
 * @author Knowledge Engineering Group
 */
public class ChiSquareFilter extends RuleFilter implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    // Threshold for chi square test. Default 0.9.
    @ConfigurableProperty
    protected double threshold = 0.9;

    /**
     * Creates a new instance of ChiSquareFilter
     */
    public ChiSquareFilter() {
        Logger.info("ChiSquareFilter");
    }

    /**
     * This will remove insignificant rules from the given rule set. Uses the preset threshold that will be mapped by LikelihoodRatio Class.
     *
     * @param rules    The rule set that is to be modified.
     * @param examples The training set.
     */
    @Override
    public TreeSet<SingleHeadRule> filterRules(final TreeSet<SingleHeadRule> rules, final de.tu_darmstadt.ke.seco.models.Instances examples) throws Exception {
        final HashSet<SingleHeadRule> filterSet = new HashSet<>();
        Iterator<SingleHeadRule> it = rules.iterator();
        while (it.hasNext()) {
            final SingleHeadRule r = it.next();
            final SingleHeadRule pred = (SingleHeadRule) r.getPredecessor();

            // filter is only applicable, if predecessor exists
            if (pred == null)
                continue;

            final de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix rStat = r.getStats();
            final de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix predStat = pred.getStats();
            final double p = rStat.getNumberOfPredictedPositive();
            final double n = rStat.getNumberOfPredictedNegative();
            final double ep = predStat.getNumberOfPredictedPositive();
            final double en = predStat.getNumberOfPredictedNegative();
            final double chi2 = chiSquareK2(p, n, ep, en);
            final double border = LikelihoodRatio.mapThreshold(threshold);

            // add insignificant rules to filter set
            if (chi2 <= border)
                filterSet.add(r);
        }

        // Remove all rules contained in the filterSet
        it = filterSet.iterator();
        while (it.hasNext()) {
            final SingleHeadRule r = it.next();
            rules.remove(r);
        }

        return rules;
    }

    /**
     * The formula for computing the chi square.
     *
     * @param p  Predicted positives.
     * @param n  Predicted negatives.
     * @param ep Predicted positives of predecessor.
     * @param en Predicted negatives of predecessor.
     * @return The chi square value.
     */
    public static double chiSquareK2(final double p, final double n, final double ep, final double en) {
        return (Math.pow(p - ep, 2) / ep) + (Math.pow(n - en, 2) / en);
    }

    @Override
    public void setProperty(final String name, final String value) {
        if (name.equals("threshold")) {
            threshold = Double.parseDouble(value);
            Logger.info("ChiSquareFilter, threshold set to: " + threshold);
        }
    }

}
