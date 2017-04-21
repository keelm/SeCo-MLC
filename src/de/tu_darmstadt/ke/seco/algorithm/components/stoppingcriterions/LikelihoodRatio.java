/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * LikelihoodRatio.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 02.11.2004, 08:17
 */

package de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions;

import java.io.Serializable;

import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.stats.TwoClassConfusionMatrix;
import weka.core.Instances;
import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.utils.Logger;

/**
 * Implemenation that will calculate the likelihood ratio. As being an implementation of IStoppingCriterion it is able to perform a check, whether that value exceeds a preset threshold.
 *
 * @author Knowledge Engineering Group
 */
public class LikelihoodRatio extends StoppingCriterion implements Serializable {

    /**
     *
     */
    private static final long serialVersionUID = 1L;

    /**
     * The Threshold that is to be used by the evaluation. Default is 0.9.
     */
    @ConfigurableProperty
    protected double m_threshold = 0.9;

    /**
     * Creates a new instance of LikelihoodRatio
     */
    public LikelihoodRatio() {
        Logger.info("LikelihoodRatio");
    }

    @Override
    public boolean checkForStop(final SingleHeadRule refinement, final Instances examples) throws Exception {
        final TwoClassConfusionMatrix stats1 = refinement.getStats();
        final double P = stats1.getNumberOfPositives();
        final double N = stats1.getNumberOfNegatives();
        final double p = stats1.getNumberOfTruePositives();
        final double n = stats1.getNumberOfFalsePositives();
        final double ep = (p + n) * (P / (P + N));
        final double en = (p + n) * (N / (P + N));
        final double lrs = evalLikelihoodRatioStatistic(p, n, ep, en);
        // m_log.trace("Likelihoodratio
        // LRS("+p+","+n+","+P+","+N+","+ep+","+en+") = "+lrs+" for rule
        // "+refinement);
        final boolean result = (lrs <= mapThreshold(m_threshold));
        if (result) {
            Logger.debug("StopCriterion matched for rule " + refinement);
        }
        return result;
    }

    /**
     * This maps the threshold.
     *
     * @param t The threshold that has to be one of: 0.7, 0.75, 0.8, 0.85, 0.9 0.95, .0.975, 0.99, 0.995
     * @return The border that has to be compared with the lrs-value.
     * @throws Exception If the border for the argument (threshold) is not defined.
     */
    public static double mapThreshold(final double t) throws Exception {
        if (t == 0) {
            return 0;
        }
        if (t == 0.7) {
            return 1.07;
        }
        if (t == 0.75) {
            return 1.32;
        }
        if (t == 0.8) {
            return 1.64;
        }
        if (t == 0.85) {
            return 2.07;
        }
        if (t == 0.9) {
            return 2.71;
        }
        if (t == 0.95) {
            return 3.84;
        }
        if (t == 0.975) {
            return 5.02;
        }
        if (t == 0.99) {
            return 6.63;
        }
        if (t == 0.995) {
            return 7.88;
        }
        throw new Exception("Unknown threshold " + t);
    }

    /**
     * Calculates the likelihood ratio statistic.
     */
    public static double evalLikelihoodRatioStatistic(final double p, final double n, final double ep, final double en) {
        return 2 * (p * Math.log(p / ep) + n * Math.log(n / en));
    }
}
