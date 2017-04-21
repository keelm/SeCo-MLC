package de.tu_darmstadt.ke.seco.multilabelrulelearning;

import de.tu_darmstadt.ke.seco.models.Condition;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule;
import de.tu_darmstadt.ke.seco.models.MultiHeadRule.Head;
import de.tu_darmstadt.ke.seco.models.MultiHeadRuleSet;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class MultiLabelPostProcessor {

    public final MultiHeadRuleSet postProcess(final MultiHeadRuleSet ruleSet) {
        int modified = 0;
        MultiHeadRuleSet result = new MultiHeadRuleSet();
        Iterator<MultiHeadRule> iterator = ruleSet.iterator();
        Map<Collection<Condition>, MultiHeadRule> rules = new HashMap<>();

        while (iterator.hasNext()) {
            MultiHeadRule rule = iterator.next();
            Head head = rule.getHead();

            if (head.size() > 0 && head.iterator().next().getAttr().name().equals("magicSkipHead")) {
                result.addRule(rule);
                rules.clear();
            } else {
                Collection<Condition> body = rule.getBody();
                MultiHeadRule existingRule = rules.get(body);

                if (existingRule == null) {
                    result.addRule(rule);
                    rules.put(body, rule);
                } else {
                    modified++;
                }

                if (existingRule != null) {
                    Head existingHead = existingRule.getHead();

                    for (Condition labelAttribute : head.getConditions()) {
                        existingHead.addCondition(labelAttribute);
                    }
                }
            }
        }

        result.setDefaultRule(ruleSet.getDefaultRule());
        result.setLabelIndices(ruleSet.getLabelIndices());
        System.out.println("Modified " + modified + " rules\n");
        return result;
    }

}