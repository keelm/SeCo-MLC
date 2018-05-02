package de.tu_darmstadt.ke.seco.multilabelrulelearning.evaluation.strategy;

import de.tu_darmstadt.ke.seco.models.MultiHeadRule;

public abstract class BoostingStrategy {

    public abstract void evaluate(MultiHeadRule rule);

    public abstract double evaluate(MultiHeadRule rule, int numberOfLabelsInTheHead);

}
