/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * FoilGain.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <p> This file
 * implements a generic class for evaluating a rule with Foil's information gain, i.e. tp*(log2(tp/(tp+fp)) -
 * log2(tp'/(tp'+fp'))) where tp' and fp' are the true and false positives of the parent rule.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class FoilGain extends GainHeuristic {

    private static final long serialVersionUID = 8576052582854137254L;

    /**
     * Constructor
     */
    public FoilGain() {
    }

    /**
     * evaluates a rule with the Foil's information gain heuristic. Rules without predecessors (getPredecessor() ==
     * null) are evaluated with 0.
     *
     * @param r The candidate rule that should be evaluated.
     * @return the information gain of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        final Rule p = r.getPredecessor();
        if (p == null)
            return 0;
        final TwoClassConfusionMatrix rs = r.getStats();

        // The implementation in JRip does neither use precision nor laplace for the computation
        // laplace would be +2 in the denominator instead of +1).
        return rs.getNumberOfTruePositives() *
                ((Math.log(this.denom(rs)) / Math.log(2)) - (Math.log(this.denom(p.getStats())) / Math.log(2)));
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        throw new UnsupportedOperationException();
    }

    private double denom(final TwoClassConfusionMatrix stats) {
        return (stats.getNumberOfTruePositives() + 1) / (stats.getNumberOfPredictedPositive() + 1);
    }

}
