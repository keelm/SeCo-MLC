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
 * Added by Frederik Janssen Created on 06.05.2008
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

/**
 * The heuristic Linear Regression is the product of a linear regression algorithm that was trained on a meta data set
 * where the goal was to predict the true precision of a candidate rule, i.e., not the regular precision computed on the
 * training set but a more realistic one computed on a hold-out validation set. The statistics of the rule that were
 * used in this research were log (tp+1), log (fp+1), log (P+1), log (N+1), tp/P, fp/N, P/(P+N) and tp/(tp+fp) <p> For
 * more information see (Janssen and F�rnkranz, Machine Learning 2010).
 *
 * @author Frederik Janssen 6.5.2008
 */
public class LinearRegression extends ValueHeuristic {

    private static final long serialVersionUID = 1540790254412061429L;
    private final TruePositiveRate truePositiveRate = new TruePositiveRate();
    private final FalsePositiveRate falsePositiveRate = new FalsePositiveRate();

    /**
     * Constructor
     */
    public LinearRegression() {
    }

    /**
     * evaluates a rule with the linear regression estimate
     *
     * @param r The candidate rule that should be evaluated.
     * @return the linear regression estimate of the rule
     */
    @Override
    public double evaluateRule(final Rule r) {
        return evaluateConfusionMatrix(r.getStats());
    }

    @Override
    public double evaluateConfusionMatrix(final TwoClassConfusionMatrix confusionMatrix) {
        // all needed statistics of the candidate rule
        final double lp = Math.log1p(confusionMatrix.getNumberOfTruePositives());
        final double ln = Math.log1p(confusionMatrix.getNumberOfFalsePositives());
        final double lP = Math.log1p(confusionMatrix.getNumberOfPositives());
        final double lN = Math.log1p(confusionMatrix.getNumberOfNegatives());
        final double p = confusionMatrix.getNumberOfTruePositives();
        final double n = confusionMatrix.getNumberOfFalsePositives();
        final double P = confusionMatrix.getNumberOfPositives();
        final double N = confusionMatrix.getNumberOfNegatives();
        final double apriori = (P + N == 0) ? 0 : P / (P + N);
        final double precTrain = (p + n == 0) ? 0 : p / (p + n);

        // the values for all features determined by the linear regression
        final double m_lP = 0.0709;
        final double m_lN = -0.0255;
        final double m_apriori = -0.0521;
        final double m_lp = 0.1139;
        final double m_ln = -0.0588;
        final double m_pP = 0.1379;
        final double m_nN = -0.3673;
        final double m_precTrain = -0.1032;
        final double m_p = 0.0;
        final double m_n = 0.0;
        final double m_P = 0.0;
        final double m_N = 0.0;
        final double m_con = 0.427;

        // True Precision
        final double predPrec =
                m_P * P + m_N * N + m_p * p + m_n * n + m_lP * lP + m_lN * lN + m_apriori * apriori + m_lp * lp +
                        m_ln * ln + m_pP * truePositiveRate.evaluateConfusionMatrix(confusionMatrix) +
                        m_nN * falsePositiveRate.evaluateConfusionMatrix(confusionMatrix) +
                        m_precTrain * precTrain + m_con;

        return (predPrec);
    }

}