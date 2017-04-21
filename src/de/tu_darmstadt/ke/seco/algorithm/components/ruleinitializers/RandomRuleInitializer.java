package de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers;

import java.io.Serializable;

import de.tu_darmstadt.ke.seco.models.SingleHeadRule;
import weka.core.Instance;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.models.Instances;
import de.tu_darmstadt.ke.seco.utils.Logger;

public class RandomRuleInitializer extends RuleInitializer implements Serializable {

	/**
	 * Serial Version UID
	 */
	private static final long serialVersionUID = 1L;

	/** Creates a new instance of RandomRuleInitializer */
	public RandomRuleInitializer() {
		Logger.info("RandomRuleInitializer used");
	}

	/**
	 * Generates a CandidateRule from a randomly chosen example
	 *
	 * @param examples
	 *            the set of examples to choose from
	 * @return the CandidateRule
	 */
	@Override
	public SingleHeadRule[] initializeRule(Heuristic heuristic, final Instances examples, final double classValue) throws Exception {
		// Randomly select one of the given examples
		final int i = random.nextInt(examples.numInstances());
		final Instance inst = examples.instance(i);

		// Transform example into a CandidateRule
		final SingleHeadRule rule = new SingleHeadRule(heuristic, inst);

		final SingleHeadRule[] result = new SingleHeadRule[1];
		result[0] = rule;

		return result;
	}

	/**
	 * returns a string representation of this class
	 */
	@Override
	public String toString() {
		return this.getClass().getName();
	}

}
