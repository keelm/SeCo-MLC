/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * BeamWidthFilter.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 03.11.2004, 12:38
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulefilters;

import java.io.Serializable;
import java.util.TreeSet;

import de.tu_darmstadt.ke.seco.algorithm.components.ConfigurableProperty;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.utils.Logger;

/**
 * This filter will reduce the rule set to the beam.
 *
 * @author Knowledge Engineering Group
 */
public class BeamWidthFilter extends RuleFilter implements Serializable {
	/**
	 *
	 */
	private static final long serialVersionUID = 1L;

	// Logger instance.

	/**
	 * The beam width for this filter. Default is 1.
	 */
	@ConfigurableProperty
	protected int beamwidth = 1;

	/** Creates a new instance of BeamWidthFilter */
	public BeamWidthFilter() {
		Logger.info("BeamWidthFilter");
	}

	/**
	 * This will remove all rules outside the preset beam width.
	 *
	 * @param rules
	 *            The rule set that is to be modified.
	 * @param examples
	 *            The training set.
	 */
	@Override
	public TreeSet<SingleHeadRule> filterRules(final TreeSet<SingleHeadRule> rules, final de.tu_darmstadt.ke.seco.models.Instances examples) throws Exception {
		final Object[] r = rules.toArray();
		for (int i = beamwidth; i < r.length; i++)
			rules.remove(r[i]);

		return rules;
	}

	@Override
	public void setProperty(final String name, final String value) {
		if (name.equals("beamwidth")) {
			final int beam = Integer.parseInt(value);
			if (beam <= 0) {
				Logger.error("beam width error");
				System.err.println("ERROR: Beamsize was " + beam + " but has to be > 0. Therefore it was set to 1!");
				beamwidth = 1;
			}
			else
				beamwidth = beam;
		}
	}

	@Override
	public String getProperty(final String name) {
		if (name.equals("beamwidth"))
			return String.valueOf(beamwidth);

		return null;
	}
}
