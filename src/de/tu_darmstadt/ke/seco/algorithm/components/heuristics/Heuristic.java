/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * SearchHeuristic.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Johannes F�rnkranz
 */

package de.tu_darmstadt.ke.seco.algorithm.components.heuristics;

import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import de.tu_darmstadt.ke.seco.models.Rule;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.averaging.AveragingStrategy;
import de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy.EvaluationStrategy;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;

import java.io.Serializable;

/**
 * The seco package implements generic functionality for simple separate-and-conquer rule learning. <p> SearchHeuristic
 * implements a generic class for search heuristics for rule learning. A Search Heuristic takes a TwoClassStats
 * confusion matrix and the predecessor rule as an argument and returns an evaluation. Most heuristics simply ignore the
 * rule, but some will use it (e.g., information gain for getting the parent gain or MDL-based heuristics for getting
 * the rule length) <p> For more details on rule learning search heuristics and their equivalences, see (F�rnkranz &
 * Flach, ICML-03)
 *
 * @author Knowledge Engineering Group
 * @version $Revision: 354 $
 */
public abstract class Heuristic extends SeCoComponent implements Serializable {

    public enum Characteristic {

        ANTI_MONOTONOUS,

        DECOMPOSABLE

    }

    private static final long serialVersionUID = 41175323637258937L;

    /**
     * computes the evaluation for a possible rule.
     *
     * @param r The candidate rule.
     * @returns a heuristic evaluation of the candidate rule
     */
    public abstract double evaluateRule(Rule r);

    /**
     * computes the evaluation for a given confusion matrix.
     *
     * @param confusionMatrix The confusion matrix
     * @return a heuristic evaluation of the confusion matrix
     */
    public abstract double evaluateConfusionMatrix(TwoClassConfusionMatrix confusionMatrix);

    /**
     * @return boolean (true, if ValueHeuristic, false if GainHeuristic)
     */
    public abstract boolean isValueHeuristic();

    public Characteristic getCharacteristic(final EvaluationStrategy evaluationStrategy,
                                            final AveragingStrategy averagingStrategy) {
        return null;
    }

}