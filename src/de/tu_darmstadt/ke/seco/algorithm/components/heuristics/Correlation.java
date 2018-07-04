/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Correlation.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <pruningDepth> This file
 * implements a generic class for evaluating a rule with the Correlation-estimate, i.e. tp*tn -
 * fp*fn/sqrt(Pos*Neg*Covered*Uncovered)
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class Correlation extends ValueHeuristic {

    private static final long serialVersionUID = 3030164309402476963L;

    /* Constructor */
    public Correlation() {
    }

    /**
     * evaluates a rule with the Correlation estimate
     *
     * @param r The candidate rule that should be evaluated.
     * @return the correlation estimate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        final double p = confusionMatrix.getNumberOfTruePositives();
        final double n = confusionMatrix.getNumberOfFalsePositives();
        final double up = confusionMatrix.getNumberOfFalseNegatives();
        final double un = confusionMatrix.getNumberOfTrueNegatives();
        final double P = p + up;
        final double N = n + un;
        final double nom = P * N * (p + n) * (up + un);
        if (Math.abs(nom - 0) < 0.0001)
            return 0;
        else
            return ((p * N - n * P) / Math.sqrt(nom));
    }

}