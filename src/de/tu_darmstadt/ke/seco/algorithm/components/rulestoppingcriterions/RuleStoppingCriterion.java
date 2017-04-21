/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * IRuleStoppingCriterion.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 29.10.2004, 17:11
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions;

import java.lang.reflect.Field;

import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import de.tu_darmstadt.ke.seco.algorithm.SeCoAlgorithm;
import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;

/**
 * An interface for objects that provide a rule stopping criterion.
 *
 * @author Knowledge Engineering Group
 */
public abstract class RuleStoppingCriterion extends SeCoComponent {

	private static final long serialVersionUID = 3053962755627300492L;

	/**
	 * Determines whether the rule stopping criterion fulfilled.
	 *
	 * @param theory
	 *            The current theory.
	 * @param rule
	 *            The new best rule that has been just evaluated, not yet in theory.
	 * @param examples
	 *            The training set.
	 * @param covered
	 *            The count of covered examples.
	 * @return true, if the criterion is fulfilled, otherwise false
	 * @throws Exception
	 */
	public abstract boolean checkForRuleStop(SingleHeadRuleSet theory, SingleHeadRule rule, Instances examples, Instances covered, double classValue, SeCoAlgorithm seCoAlgorithm) throws Exception;

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
