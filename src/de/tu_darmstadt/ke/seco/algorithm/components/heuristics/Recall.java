/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * FMeasure.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Frederik Janssen
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.RuleDependentEvaluation;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

public class Recall extends ValueHeuristic {

	private static final long serialVersionUID = 1L;
	
	private final TruePositiveRate truePositiveRate = new TruePositiveRate();

    /**
     * empty Constructor
     */
    public Recall() {

    }

    /**
     * computes the F-Measure of the rule
     *
     * @param r The candidate rule that should be evaluated.
     * @return the F-Measure of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {    	
    	return truePositiveRate.evaluateConfusionMatrix(confusionMatrix);
    }

    @Override
    public Characteristic getCharacteristic(final EvaluationStrategy evaluationStrategy,
                                            final AveragingStrategy averagingStrategy) {
        if (evaluationStrategy instanceof RuleDependentEvaluation) {
            return Characteristic.DECOMPOSABLE;
        } else {
            return Characteristic.ANTI_MONOTONOUS;
        }
    }

}