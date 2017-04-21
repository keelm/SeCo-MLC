/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * IRuleFilter.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 30.10.2004, 16:36
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulefilters;

import java.util.TreeSet;

import de.tu_darmstadt.ke.seco.algorithm.components.SeCoComponent;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;

/**
 * The implementation of a rule filter as it will be usually called after Refinement.
 *
 * @author Knowledge Engineering Group
 */
public abstract class RuleFilter extends SeCoComponent {

    private static final long serialVersionUID = -3056492545396965158L;

    /**
     * This remove the filtered rules from the given TreeSet.
     *
     * @param rules    The rule set that is to be filtered.
     * @param examples The training set.
     * @throws Exception
     */
    public abstract TreeSet<SingleHeadRule> filterRules(TreeSet<SingleHeadRule> rules, Instances examples) throws Exception;
}
