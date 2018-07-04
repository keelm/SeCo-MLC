/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * RelativeLinearCost.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Frederik Janssen Created on 06.05.2008
 */
package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <pruningDepth> <pruningDepth> This file
 * implements a generic class for evaluating a rule with the relative cost measure. This is an extension to the linear
 * cost where the covered positive and negative examples are normalized by the total numbers of them. Thus it can be
 * viewed like a parametrized weighted relative accuracy heuristic. <pruningDepth> It is computed by (c*tp/P - (1-c)*fp/N) where 0
 * <= c <= 1. <pruningDepth> The default value of c is 0.342, which has been derived empirically in (Janssen and F�rnkranz, Machine
 * Learning 2010).
 *
 * @author Knowledge Engineering Group
 */
public class RelativeLinearCost extends ValueHeuristic {

    private static final long serialVersionUID = -1522835769414760411L;

    @ConfigurableProperty
    private double c = 0.342;

    /**
     * empty Constructor
     */
    public RelativeLinearCost() {

    }

    /**
     * Constructor
     *
     * @param c The parameter of the Relative Cost Measure
     * @throws Exception
     */
    public RelativeLinearCost(final double c) throws Exception {
        if (c < 0 || c > 1) {
            throw new Exception("RelativeLinearCosts parameter: 0 <= c <= 1!");
        } else {
            this.c = c;
        }
    }

    /**
     * computes the relative linear cost of the rule
     *
     * @param r The candidate rule that should be evaluated.
     * @return the relative linear cost of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        return (confusionMatrix.getNumberOfTruePositives() * c / confusionMatrix.getNumberOfPositives() -
                (1 - c) * confusionMatrix.getNumberOfFalsePositives() / confusionMatrix.getNumberOfNegatives());
    }

}