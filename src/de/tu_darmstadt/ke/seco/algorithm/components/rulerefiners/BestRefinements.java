package de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners;

import java.util.Collections;
import java.util.TreeSet;

import de.tu_darmstadt.ke.seco.models.SingleHeadRule;

public class BestRefinements extends TreeSet<SingleHeadRule> {
	/**
     * 
     */
	private static final long serialVersionUID = 1L;

	// TODO: why is numRefinements final? Shouldn't it be a configurable property?
	private final int numRefinements;

	public BestRefinements(final int n) {
		super(Collections.reverseOrder());
		this.numRefinements = n;
	}

	@Override
	public boolean add(final SingleHeadRule rule) {
		if (this.size() < numRefinements)
			return super.add(rule);
		else {
			final SingleHeadRule lowest = this.last();
			if (rule.compareTo(lowest) > 0) {
				this.remove(lowest);
				return super.add(rule);
			}
		}

		return false;
	}

}
