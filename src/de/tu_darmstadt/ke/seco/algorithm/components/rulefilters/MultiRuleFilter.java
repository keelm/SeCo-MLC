/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * MultiRuleFilter.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 * 
 * Added by Matthias Thiel Created on 17.11.2004, 16:22
 */

package de.tu_darmstadt.ke.seco.algorithm.components.rulefilters;

import java.io.Serializable;
import java.util.TreeSet;

import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;

/**
 * This filter will combine multiple other filters. The filters will be applied in a row.
 *
 * @author Knowledge Engineering Group
 */
// TODO: When the MultiRuleFilter should be configured in a configuration file, how does this look like?
public class MultiRuleFilter extends RuleFilter implements Serializable {

    /**
     *
     */
    private static final long serialVersionUID = 1L;

    /**
     * A list of the filters.
     */
    private TreeSet<RuleFilter> ruleFilters = new TreeSet<RuleFilter>();

    /**
     * Creates a new instance of MultiRuleFilter
     */
    public MultiRuleFilter() {

    }

    /**
     * This will apply all added filters on the rule set.
     *
     * @param rules    the rule set.
     * @param examples The training set.
     */
    @Override
    public TreeSet<SingleHeadRule> filterRules(final TreeSet<SingleHeadRule> rules, final Instances examples) throws Exception {
        for (final RuleFilter ruleFilter : ruleFilters)
            ruleFilter.filterRules(rules, examples);

        return rules;
    }

    /**
     * This will add another filter. The filters will be applied in the order in which they will be added by this method.
     *
     * @param ruleFilter The new filter.
     */
    // TODO by m.zopf: this method is never used. so the multirulefilter is useless.
    public void addFilter(final RuleFilter ruleFilter) throws Exception {
        ruleFilters.add(ruleFilter);
    }
}
