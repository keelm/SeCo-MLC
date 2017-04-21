/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * WeightPerIteration.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Marc Ruppert Created on 29.06.2006
 */

package de.tu_darmstadt.ke.seco.algorithm.components.weightmodels;

import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.utils.Logger;
import weka.core.Instance;

import java.io.Serializable;

/**
 * This is a Weighter, that multiplicates the weight of each covered example by m_multWeightfactor
 * 
 * Allmost sure it will result in a endless loop if the RuleStop doesn't consider the covering!
 * 
 * @author Knowledge Engineering Group
 */
public class WeightPerIteration extends WeightModel implements Serializable {

	/**
     * 
     */
	private static final long serialVersionUID = 1L;

	/**
	 * A factor for multiplicative increase/decrease of weight
	 */
	protected double m_multWeightfactor = 0.5;

	/**
	 * A factor for additve increase/decrease of weight
	 */
	protected double m_addWeightfactor = 0.0;

	public WeightPerIteration() {
		Logger.info("WeightPerIteration used");
	}

	/**
	 * Changes the weight of each example to the multiplicate of m_multWeightfactor
	 * 
	 * @param examples
	 *            The Examples where the weight have to be changed
	 * @param covered
	 *            This is a container for counting how many rules cover a example
	 * @param rule
	 *            The rule, that initiate the "covering" and so the new weights
	 * 
	 * @return A array of the weighted examples (0) and the cover counter (1)
	 * @throws Exception
	 */
	@Override
	public Instances[] changeWeights(final Instances examples, final Instances covered, final SingleHeadRule rule) throws Exception {
		// double oldweight;
		double newweight;
		double coverCount = 0.0;
		Instance coverTest;
		for (int i = 0; i < examples.numInstances(); i++) {
			coverTest = examples.instance(i);
			if (rule.covers(coverTest)) { // if the SingleHeadRule covers this Instance,
											// change weight
				coverCount = covered.instance(i).weight() + 1;
				// call the updateWeight method
				newweight = calcNewWeight(examples.instance(i), covered.instance(i));
				// oldweight = examples.instance(i).weight();
				// examples.instance(i).setWeight(oldweight*m_multWeightfactor);
				// set the new weight
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
		final double oldweight = example.weight();
		final double newweight = oldweight * m_multWeightfactor;
		return newweight;
	}

	/**
	 * This will be used for setting properties.
	 * 
	 * @param name
	 *            The name of the property.
	 * @param value
	 *            The value of the property.
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
