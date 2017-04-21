package de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners;

import java.io.Serializable;

import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import de.tu_darmstadt.ke.seco.models.SingleHeadRuleSet;

/**
 * SingleHeadRule Refiner class that refines rules using a bidirectional strategy, i.e. it can either add or delete conditions to/from a rule
 *
 * @author AnneChristineKarpf
 *
 */
public class BidirectionalRefiner extends RuleRefiner implements Serializable {

	private static final long serialVersionUID = 1L;

	/**
	 * Refiner which computes the specializations of a rule
	 */
	private final TopDownRefiner specializer;

	/**
	 * Refiner which computes the generalizations of a rule
	 */
	private final BottomUpRefiner generalizer;

	public BidirectionalRefiner() {
		specializer = new TopDownRefiner();
		generalizer = new BottomUpRefiner();
	}

	@Override
	public SingleHeadRuleSet refineRule(final SingleHeadRule c, final Instances examples, final double classValue) throws Exception {

		// Make sure that m_heuristic is set
		if (heuristic == null)
			heuristic = c.getHeuristic();

		// Delete all rules from refinements
		clearRefinements();

		// Compute generalizations and specializations
		final SingleHeadRuleSet specializations = specializer.refineRule(c, examples, classValue);
		final SingleHeadRuleSet generalizations = generalizer.refineRule(c, examples, classValue);

		/*
		 * Things to keep in mind at this point: - The rules must be evaluated before being added to refinements. This method assumes that the specializer and generalizer take care of that. - The abstract superclass AbstractRefiner makes sure that refinements is of the type BestRefinements if a beamwidth is given, so this is not checked here.
		 */
		final SingleHeadRuleSet allRefinements = new SingleHeadRuleSet();

		for (final SingleHeadRule rule : specializations)
			allRefinements.addRule(rule);

		for (final SingleHeadRule rule : generalizations)
			allRefinements.addRule(rule);

		return allRefinements;
	}

	@Override
	public void setProperty(final String name, final String value) {
		// Save values
		super.setProperty(name, value);

		// Propagate values to components
		specializer.setProperty(name, value);
		generalizer.setProperty(name, value);

	}

}