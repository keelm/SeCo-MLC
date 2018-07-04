/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
/*
 * MaxPosCover.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Marc Ruppert Created on 8. Juni 2006, 11:14
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning.
 * <pruningDepth>
 * This file implements a generic class for evaluating a rule with the number of covered positive examples
 * <pruningDepth>
 * This is outsourced to support other than the standard heuristics for AQR
 *
 * @author Knowledge Engineering Group
 */
public class MaxPosCover extends ValueHeuristic {

    private static final long serialVersionUID = 2070238113593026775L;

    /**
     * Creates a new instance of MaxPosCover
     */
    public MaxPosCover() {
    }

    /**
     * evaluates a rule with the max pos estimate
     *
     * @param r The candidate rule that should be evaluated.
     * @return the max pos estimate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        return confusionMatrix.getNumberOfTruePositives();
    }

}