/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * WRAcc.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <pruningDepth> <pruningDepth> This file
 * implements a generic class for evaluating a rule with weighted relative accuracy. weighted relative accuracy is
 * (tp+fp)/total (tp/(tp+fp) - prior) where total is tp+fp+fn+tn and prior is the prior probability (tp+fn)/total. <pruningDepth>
 * <pruningDepth> You may also want to consider the equivalent but faster RateDiff. For more details on the the equivalences
 * between search heuristics see (F�rnkranz & Flach, ICML-03).
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class WRAcc extends ValueHeuristic {

    private static final long serialVersionUID = -2750661086054304080L;

    /**
     * Constructor
     */
    public WRAcc() {
    }

    /**
     * computes the weighted relative accuracy of the rule
     *
     * @param r The candidate rule that should be evaluated.
     * @return the weighted relative accuracy of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        final double total = confusionMatrix.getNumberOfExamples();
        return confusionMatrix.getNumberOfTruePositives() / total -
                confusionMatrix.getNumberOfPredictedPositive() / total * confusionMatrix.getNumberOfPositives() / total;
    }

}