/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * LinearCosts.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <pruningDepth> <pruningDepth> Linear Costs
 * implements a class for evaluating a rule with a linear cost function (c*tp - (1-c)*fp), where 0 <= c <= 1. <pruningDepth> <pruningDepth> c
 * = 1 means that only covering positives counts, c = 0 means that only excluding negatives counts, values in between
 * trade off between these two extremes. The default value of c is 0.437, which has been derived empirically in (Janssen
 * and F�rnkranz, Machine Learning 2010). <pruningDepth> <pruningDepth> The parameter can be canged via setProperty.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */

public class LinearCost extends ValueHeuristic {

    private static final long serialVersionUID = 7280490141832662538L;

    @ConfigurableProperty
    private double c = 0.437;

    /**
     * Empty constructor, c will be set to 0.437.
     */
    public LinearCost() {

    }

    /**
     * Constructor.
     *
     * @param c The value for the cost trade-off.
     * @throws Exception unless 0 <= c <= 1.
     */
    public LinearCost(final double c) throws Exception {
        if (c < 0 || c > 1) {
            throw new Exception("LinearCosts parameter: 0 <= c <= 1!");
        } else {
            this.c = c;
        }
    }

    /**
     * computes the linear cost estimate of the rule
     *
     * @param r The candidate rule that should be evaluated.
     * @return the linear cost estimate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        return c * confusionMatrix.getNumberOfTruePositives() - (1 - c) * confusionMatrix.getNumberOfFalsePositives();
    }

}
