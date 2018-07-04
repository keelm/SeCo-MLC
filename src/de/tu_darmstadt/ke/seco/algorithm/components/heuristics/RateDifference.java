/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * RateDiff.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <pruningDepth> This file
 * implements a generic class for evaluating a rule with the difference of the true positive rate and the true negative
 * rate. <pruningDepth> For rules with the same example distribution ((tp + fn) and (fp + tn) are constant), this is eqivalent to
 * weighted relative accuracy (WRAcc), but presumably faster.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class RateDifference extends ValueHeuristic {

    private static final long serialVersionUID = -6305586234582024334L;
    private final TruePositiveRate truePositiveRate = new TruePositiveRate();
    private final FalsePositiveRate falsePositiveRate = new FalsePositiveRate();

    /**
     * Constructor
     */
    public RateDifference() {
    }

    /**
     * evaluates a rule with the difference between the true positive and the false positive rate
     *
     * @param r The candidate rule that should be evaluated.
     * @return the difference between the true positive and the false positive rate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        return (truePositiveRate.evaluateConfusionMatrix(confusionMatrix) -
                falsePositiveRate.evaluateConfusionMatrix(confusionMatrix));
    }

}