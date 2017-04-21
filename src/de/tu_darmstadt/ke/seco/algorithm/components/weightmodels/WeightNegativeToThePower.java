/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * WeightNegativeToThePower.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Marc Ruppert Created on 04.07.2006 Modified by Viktor Seifert
 */

package de.tu_darmstadt.ke.seco.algorithm.components.weightmodels;

import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.utils.Logger;
import weka.core.Instance;

import java.io.Serializable;

/**
 * This is a Weighter, that will set the Weight for covered negatives up. Based on the formula form "Lightweigt SingleHeadRule Induction" from Weiss & Indurkhya Where e(i) is the cumulative error for case i. The Meaning: if i have 10 rules, and 4 say that i is a no, but 6 say it is yes, and i is a yes, then the errorweight for i is 1+(4^multfactor) (aka 1 + 4^3 = 65) This weighing doesn't work with the normal AQR, cause in the rules of AQR are no negatives covered!
 * 
 * @author Knowledge Engineering Group
 */
public class WeightNegativeToThePower extends WeightModel implements Serializable {

	/**
     * 
     */
	private static final long serialVersionUID = 1L;

	/**
	 * A factor for multiplicative increase/decrease of weight
	 */
	protected double m_multWeightfactor = 3.0;

	/**
	 * A factor for additve increase/decrease of weight
	 */
	protected double m_addWeightfactor = 0.0;

	public WeightNegativeToThePower() {
		Logger.info("WeightNegativeToThePower used");
	}

	/**
	 * Changes the weight of each example to the power of m_multWeightfactor
	 * 
	 * @param examples
	 *            The Examples where the weight have to be changed
	 * @param covered
	 *            This is a container for counting how many rules cover a example
	 * @param rule
	 *            The rule, that initiate the "covering" and so the new weights
	 * 
	 * @return The weighted examples
	 * @throws Exception
	 */
	@Override
	public Instances[] changeWeights(final Instances examples, final Instances covered, final SingleHeadRule rule) throws Exception {
		// double oldweight;
		double newweight;
		Instance coverTest;
		double coverCount = 0.0;

		for (int i = 0; i < examples.numInstances(); i++) {
			coverTest = examples.instance(i);
			// if the SingleHeadRule covers this Instance and Instance is Negative, change
			// weight
			if ((rule.covers(coverTest)) && (coverTest.classValue() != rule.getPredictedValue())) {
				newweight = calcNewWeight(examples.instance(i), covered.instance(i));
				coverCount = covered.instance(i).weight() + 1;
				// oldweight = covered.instance(i).weight();
				// # of covered rules to the power of m_multWeightfactor
				// oldweight = Math.pow(coverCount,m_multWeightfactor);
				// covered.instance(i).setWeight(oldweight);
				examples.instance(i).setWeight(newweight);
				covered.instance(i).setWeight(coverCount);
			}
		}
		final Instances[] container = new Instances[2];
		container[0] = examples;
		container[1] = covered;
		return container;
	}

	/**
	 * The calcNewWeight method calculates the weight of the delivered example
	 * 
	 * @param example
	 *            The example where the weight have to be updated
	 * @param cover
	 *            The container-example with the count of covered rules
	 * 
	 * @return The weight of the Example
	 * 
	 * @throws Exception
	 * @author Marc Ruppert
	 */
	@Override
	public double calcNewWeight(final Instance example, final Instance cover) throws Exception {
		final double coverCount = cover.weight() + 1;
		double oldweight = cover.weight();
		// # of covered rules to the power of m_multWeightfactor
		oldweight = Math.pow(coverCount, m_multWeightfactor);
		return oldweight;
	}

	/**
	 * This will be used for setting properties.
	 * 
	 * @param name
	 *            The name of the property.
	 * @param value
	 *            The value of the property.
	 * 
	 *            The MultiValue here ist for the oldweight to the power of MultiValue, default (3)
	 */
	@Override
	public void setProperty(final String name, final String value) {
		if (name.equalsIgnoreCase("MultiValue"))
			m_multWeightfactor = Double.parseDouble(value);
		else if (name.equalsIgnoreCase("AddValue"))
			m_addWeightfactor = Double.parseDouble(value);
	}

	/**
	 * @return Returns the m_addWeightfactor.
	 */
	public double getM_addWeightfactor() {
		return m_addWeightfactor;
	}

	/**
	 * @param weightfactor
	 *            The m_addWeightfactor to set.
	 */
	public void setM_addWeightfactor(final double weightfactor) {
		m_addWeightfactor = weightfactor;
	}

	/**
	 * @return Returns the m_multWeightfactor.
	 */
	public double getM_multWeightfactor() {
		return m_multWeightfactor;
	}

	/**
	 * @param weightfactor
	 *            The m_multWeightfactor to set.
	 */
	public void setM_multWeightfactor(final double weightfactor) {
		m_multWeightfactor = weightfactor;
	}
}