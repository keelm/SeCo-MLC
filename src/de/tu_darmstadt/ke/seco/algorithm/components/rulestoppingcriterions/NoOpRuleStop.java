/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * NoOpRuleStop.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 17. November 2004, 19:14 Modified by Frederik Janssen
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions;

import java.io.Serializable;
import java.lang.reflect.Field;

import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;
import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.models.Instances;

/**
 * The NoRuleStop just returns FALSE.
 *
 * @author Knowledge Engineering Group
 */
public class NoOpRuleStop extends RuleStoppingCriterion implements Serializable {

	/**
	 * default serial UID
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Creates a new instance of NoRuleStop
	 */
	public NoOpRuleStop() {
	}

	/**
	 * Returns FALSE..
	 *
	 * @param theory
	 *            The current theory.
	 * @param rule
	 *            The CandidateRule that is to be compared.
	 * @param examples
	 *            The training set.
	 * @return true, if it should stop, false otherwise.
	 */
	@Override
	public boolean checkForRuleStop(final SingleHeadRuleSet theory, final SingleHeadRule rule, final Instances examples, final Instances covered, final double classValue, final SeCoAlgorithm seCoAlgorithm) throws Exception {
		return false;
	}

	@Override
	public String toString() {
		return (new ReflectionToStringBuilder(this, ToStringStyle.SHORT_PREFIX_STYLE) {
			@Override
			protected boolean accept(final Field f) {
				return false; // exclude all fields from the toString
			}
		}).toString().replaceAll("\\[\\]", "");
	}
}
