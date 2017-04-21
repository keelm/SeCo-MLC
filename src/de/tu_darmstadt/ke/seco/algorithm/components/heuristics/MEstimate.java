/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * MEstimate.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <p> <p> This file
 * implements a generic class for evaluating a rule with the m-estimate, i.e. (tp+m*prior)/(tp+fp+m). The prior
 * probability is (tp + fn) / (tp + fp + fn +tn). If you don't want to recompute this every time around, better use the
 * GeneralizedM. <p> <p> The default value of m is 22.466 as determined in (Janssen and F�rnkranz, 2010). It can be
 * canged via setProperty.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class MEstimate extends ValueHeuristic {

    private static final long serialVersionUID = 667693661510875887L;

    @ConfigurableProperty
    private double m = 22.466;

    /**
     * Empty constructor, m will be set to 22.466, the best parameter determined in (Janssen and F�rnkranz, 2010).
     */
    public MEstimate() {

    }

    /**
     * Constructor
     *
     * @param m The value of m.
     */
    public MEstimate(final double m) {
        this.m = m;
    }

    /**
     * evaluates a rule with the m-estimate
     *
     * @param r The candidate rule that should be evaluated.
     * @return the m-estimate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        return (confusionMatrix.getNumberOfTruePositives() +
                m * confusionMatrix.getNumberOfPositives() / confusionMatrix.getNumberOfExamples()) /
                (confusionMatrix.getNumberOfPredictedPositive() + m);
    }

}