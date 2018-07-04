/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * KloesgenMeasure.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Frederik Janssen
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <p> <p> This file
 * implements a generic class for evaluating a rule with the kloesgen-estimate. It is defined by (coverage)^omega *
 * (precision - apriori) where coverage = (tp + fp) / (P + N), precision = tp / (tp + fp) and apriori = P / (P + N). <p>
 * The default value of n is 0.4323 as determined in (Janssen and F�rnkranz, Machine Learning 2010). It can be changed
 * via setProperty.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class KloesgenMeasure extends ValueHeuristic {

    private static final long serialVersionUID = -3441131947943923128L;

    @ConfigurableProperty
    private double omega = 0.4323;

    private final Precision precision = new Precision();

    /**
     * empty Constructor
     */
    public KloesgenMeasure() {

    }

    /**
     * Constructor
     *
     * @param omega The parameter of the Kloesgen-Measure
     * @throws Exception
     */

    public KloesgenMeasure(final double omega) throws Exception {
        if (omega == 0) {
            throw new Exception("n = 0!!");
        } else {
            this.omega = omega;
        }
    }

    /**
     * computes the Kloesgen-Measure of the rule
     *
     * @param r The candidate rule that should be evaluated.
     * @return the Kloesgen-Measure of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        final double P = confusionMatrix.getNumberOfPositives();
        final double c = confusionMatrix.getNumberOfPredictedPositive() / confusionMatrix.getNumberOfExamples();
        final double g = precision.evaluateConfusionMatrix(confusionMatrix) - P / confusionMatrix.getNumberOfExamples();
        return (Math.pow(c, omega) * g);
    }

}