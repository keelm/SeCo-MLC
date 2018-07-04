/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * GeneralizedM.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <p> This file
 * implements a generic class for evaluating a rule with the generalized m-estimate, i.e. (tp+m*c)/(tp+fp+m).<br> c may
 * be interpreted as a general linear cost factor, just like in LinearCosts.<br> m is the m-value as in the m-heuristic.
 * It may be viewed as a trade-off between LinearCost (which assumes a cost value c) and Precision (which does not make
 * any cost assumptions). <br> If m is NaN, it will be interpreted as infinity and the LinearCosts heuristic will be
 * called. If m is 0, you get Precision.<br> The default values are m = 2 and c = 0.5, which results in the Laplace
 * heuristic. <p> <p> See (F�rnkranz & Flach, ICML-03) for details on the Generalized MEstimate.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class GeneralizedM extends ValueHeuristic {

    private static final long serialVersionUID = 1917528903112420449L;
    @ConfigurableProperty
    private double m = 2.0;
    @ConfigurableProperty
    private double c = 0.5;

    /**
     * Empty constructor, c will be set to 0.5 and m to 2.0.
     */
    public GeneralizedM() {

    }

    /**
     * Constructor
     *
     * @param m The value for m (0 <= m <= NaN, default 1).
     * @param c The cost factor (0 <= c <= 1, default 0.5).
     * @throws Exception
     */
    public GeneralizedM(final double m, final double c) throws Exception {
        this.m = m;
        if (c < 0 || c > 1) {
            throw new Exception("GeneralizedM: 0 <= c <= 1!");
        } else {
            this.c = c;
        }

    }

    /**
     * evaluates a rule with the generalized m-estimate
     *
     * @param r The candidate rule that should be evaluated.
     * @return the generalized m-estimate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        if (Double.isNaN(m)) {
            return c * confusionMatrix.getNumberOfTruePositives() -
                    (1 - c) * confusionMatrix.getNumberOfFalsePositives();
        } else {
            return (confusionMatrix.getNumberOfTruePositives() + (m * c)) /
                    (confusionMatrix.getNumberOfPredictedPositive() + m);
        }
    }

}