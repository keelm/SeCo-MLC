/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Laplace.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning.
 * <p>
 * This file implements a generic class for evaluating a rule with the Laplace-estimate, i.e. (tp+1)/(tp+fp+2)
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class Laplace extends ValueHeuristic {

    private static final long serialVersionUID = -4436764346373544659L;

    /**
     * Constructor
     */
    public Laplace() {
    }

    /**
     * computes the Laplace estimate of the rule
     *
     * @param r The candidate rule that should be evaluated.
     * @return the Laplace estimate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        return (confusionMatrix.getNumberOfTruePositives() + 1) / (confusionMatrix.getNumberOfPredictedPositive() + 2);
    }

}