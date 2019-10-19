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

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.ExampleBasedAveraging;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.RuleDependentEvaluation;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <p> <p> This file
 * implements a generic class for evaluating a rule with the F-Measure. The F-Measure origins from the Information
 * Retrieval community and is defined by (beta^2+1)*precision*recall / (beta^2)*precision+recall where precision = tp /
 * (tp + fp) and recal = tp / P. In (Janssen and F�rnkranz, Machine Learning 2010) an optimal parameter setting for beta
 * was determined and if beta is not given it is initialized to this derived best setting. It can be changed via
 * setProperty.
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public class FMeasure extends ValueHeuristic {

    private static final long serialVersionUID = -3541563713409740796L;

    @ConfigurableProperty
    private double beta = 0.5;

    private final Precision precision = new Precision();
    private final TruePositiveRate truePositiveRate = new TruePositiveRate();

    /**
     * empty Constructor
     */
    public FMeasure() {

    }

    /**
     * Constructor
     *
     * @param beta The parameter of the F-Measure
     * @throws Exception
     */
    public FMeasure(final double beta) throws Exception {
        this.beta = beta;
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
        if ((precision.evaluateConfusionMatrix(confusionMatrix) +
                truePositiveRate.evaluateConfusionMatrix(confusionMatrix)) == 0) {
            return 0;
        }
        final double a = (Math.pow(beta, 2) + 1) * truePositiveRate.evaluateConfusionMatrix(confusionMatrix) *
                precision.evaluateConfusionMatrix(confusionMatrix);
        return a / (Math.pow(beta, 2) * precision.evaluateConfusionMatrix(confusionMatrix) +
                truePositiveRate.evaluateConfusionMatrix(confusionMatrix));
    }

    /*
     * calculate FMeasure from two matrizes, because precision and recall are calculated separately with different counting schemes
     */
    
    public double evaluateMixedConfusionMatrix(final TwoClassConfusionMatrix precisionMatrix, final TwoClassConfusionMatrix recallMatrix) {
    	if ((precision.evaluateConfusionMatrix(precisionMatrix) +
                truePositiveRate.evaluateConfusionMatrix(recallMatrix)) == 0) {
            return 0;
        }
        final double a = (Math.pow(beta, 2) + 1) * truePositiveRate.evaluateConfusionMatrix(recallMatrix) *
                precision.evaluateConfusionMatrix(precisionMatrix);
        return a / (Math.pow(beta, 2) * precision.evaluateConfusionMatrix(precisionMatrix) +
                truePositiveRate.evaluateConfusionMatrix(recallMatrix));
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