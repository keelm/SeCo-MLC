package de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners;

import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;

import java.io.Serializable;

/**
 * SingleHeadRule Refiner class that refines rules by deleting conditions
 *
 * @author AnneChristineKarpf
 */
public class BottomUpRefiner extends RuleRefiner implements Serializable {

    /**
     * serial Version UID
     */
    private static final long serialVersionUID = 1L;

    @Override
    public SingleHeadRuleSet refineRule(final SingleHeadRule c, final Instances examples, final double classValue) throws Exception {

        // Make sure that m_heuristic is set
        if (heuristic == null)
            heuristic = c.getHeuristic();

        // Delete all rules from refinements
        clearRefinements();

        // if c contains no conditions, it cannot be generalized
        // => return empty set
        if (c.length() < 1)
            return new SingleHeadRuleSet();

        // loop through all conditions in the rule and create generalizations by
        // deleting the i-th condition, respectively.
        for (int i = 0; i < c.length(); i++) {
            final SingleHeadRule newRule = (SingleHeadRule) c.generalize(i);

            // Begin Copy-Paste from TopDownRefiner
            if (beamwidth != null)
                evaluateRule(newRule, examples, classValue);

            addToRefinements(newRule);
        }

        // return the list of generalizations of c
        return getRefinementsAsRuleSet();
    }

    @Override
    public void setProperty(final String name, final String value) {
        super.setProperty(name, value);
    }

}
