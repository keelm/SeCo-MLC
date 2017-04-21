/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * AqrDifference.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Marc Ruppert Created on 27.06.2006
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <p> This file
 * implements a generic class for evaluating a rule with the difference between the covered positive and negative
 * examples, i.e., tp - fp. <p> <p> Note, however, that for rules with the same example distribution (tp + fn) and (fp +
 * tn) are constant), this is eqivalent to accuracy, but presumbably somewhat faster. <p> It is a spezial version for
 * AQR
 *
 * @author Knowledge Engineering Group
 */
public class AqrDifference extends ValueHeuristic {

    private static final long serialVersionUID = 7210028096383496023L;

    /**
     * Constructor
     */
    public AqrDifference() {
    }

    /**
     * @param r The candidate rule that should be evaluated.
     * @return the difference between the number of positive and negative examples covered by the rule. This is
     * equivalent but presumably faster than accuracy.
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        final double P = confusionMatrix.getNumberOfPositives();
        final double N = confusionMatrix.getNumberOfNegatives();
        final double p = confusionMatrix.getNumberOfTruePositives();
        final double n = confusionMatrix.getNumberOfFalsePositives();

        if (p < 1.0)
            return ((P + N) * (-1));
        // regular case
        return (p - n);
    }

}