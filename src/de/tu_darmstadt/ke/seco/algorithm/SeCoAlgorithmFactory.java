package de.tu_darmstadt.ke.seco.algorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Hashtable;

import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserException;
import org.xmlpull.v1.XmlPullParserFactory;

import de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors.CandidateSelector;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.algorithm.components.postprocessors.PostProcessor;
import de.tu_darmstadt.ke.seco.algorithm.components.rulefilters.RuleFilter;
import de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers.RuleInitializer;
import de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners.RuleRefiner;
import de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions.RuleStoppingCriterion;
import de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions.StoppingCriterion;
import de.tu_darmstadt.ke.seco.utils.Logger;

public class SeCoAlgorithmFactory {

    public static String COMPONENTS_PATH = "de.tu_darmstadt.ke.seco.algorithm.components";
    public static String CANDIDATE_SELECTORS_PATH = COMPONENTS_PATH + ".candidateselectors";
    public static String HEURISTICS_PATH = COMPONENTS_PATH + ".heuristics";
    public static String POST_PROCESSORS_PATH = COMPONENTS_PATH + ".postprocessors";
    public static String RULE_FILTERS_PATH = COMPONENTS_PATH + ".rulefilters";
    public static String RULE_INITIALIZERS_PATH = COMPONENTS_PATH + ".ruleinitializers";
    public static String RULE_REFINERS_PATH = COMPONENTS_PATH + ".rulerefiners";
    public static String RULE_STOPPING_CRITERIONS_PATH = COMPONENTS_PATH + ".rulestoppingcriterions";
    public static String STOPPING_CRITERIONS_PATH = COMPONENTS_PATH + ".stoppingcriterions";

    public static SeCoAlgorithm buildAlgorithmFromFile(final String filename) throws XmlPullParserException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException, ClassNotFoundException, IOException {
        Logger.info("Building SeCoAlgorithm from file '" + filename + "'.");
        final XmlPullParserFactory xmlPullParserFactory = XmlPullParserFactory.newInstance(System.getProperty(XmlPullParserFactory.PROPERTY_NAME), null);
        final XmlPullParser xmlPullParser = xmlPullParserFactory.newPullParser();
        xmlPullParser.setInput(new BufferedReader(new FileReader(filename)));

        return buildAlgorithmFromXMLPullParser(filename, xmlPullParser);
    }

    private static SeCoAlgorithm buildAlgorithmFromXMLPullParser(final String filename, final XmlPullParser xmlPullParser) throws XmlPullParserException, IOException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException, ClassNotFoundException {
        final SeCoAlgorithm seCoAlgorithm = new SeCoAlgorithm(filename);
        xmlPullParser.next(); // skip XmlPullParser.START_DOCUMENT node
        int eventType = xmlPullParser.getEventType();

        if (xmlPullParser.getName().equals("seco"))
            seCoAlgorithm.setAttributes(getProperties(xmlPullParser));
        else
            throw new XmlPullParserException("XML configuration file has to start with seco node.");

        while ((eventType = xmlPullParser.next()) != XmlPullParser.END_DOCUMENT)
            if (eventType == XmlPullParser.START_TAG)
                processSeCoCompenentNode(xmlPullParser, seCoAlgorithm);

        return seCoAlgorithm;
    }

    private static void processSeCoCompenentNode(final XmlPullParser xmlPullParser, final SeCoAlgorithm seCoAlgorithm) throws InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException, ClassNotFoundException {
        final String nodeName = xmlPullParser.getName();

        final Hashtable<String, String> properties = getProperties(xmlPullParser);
        final String classname = properties.get("classname");
        properties.remove("classname");

        if (nodeName.equals("candidateselector") || nodeName.equals("cs")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".candidateselectors." + classname;
            final CandidateSelector candidateSelector = (CandidateSelector) createByClassname(fullyQualifiedClassName);
            candidateSelector.setProperties(properties);
            candidateSelector.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setCandidateSelector(candidateSelector);
        } else if (nodeName.equals("heuristic") || nodeName.equals("h")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".heuristics." + classname;
            final Heuristic heuristic = (Heuristic) createByClassname(fullyQualifiedClassName);
            heuristic.setProperties(properties);
            heuristic.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setHeuristic(heuristic);
        } else if (nodeName.equals("postprocessor") || nodeName.equals("pp")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".postprocessors." + classname;
            final PostProcessor postProcessor = (PostProcessor) createByClassname(fullyQualifiedClassName);
            postProcessor.setProperties(properties);
            postProcessor.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setPostProcessor(postProcessor);
        } else if (nodeName.equals("rulefilter") || nodeName.equals("rf")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".rulefilters." + classname;
            final RuleFilter ruleFilter = (RuleFilter) createByClassname(fullyQualifiedClassName);
            ruleFilter.setProperties(properties);
            ruleFilter.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setRuleFilter(ruleFilter);
        } else if (nodeName.equals("ruleinitializer") || nodeName.equals("ri")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".ruleinitializers." + classname;
            final RuleInitializer ruleInitializer = (RuleInitializer) createByClassname(fullyQualifiedClassName);
            ruleInitializer.setProperties(properties);
            ruleInitializer.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setRuleInitializer(ruleInitializer);
        } else if (nodeName.equals("rulerefiner") || nodeName.equals("rr")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".rulerefiners." + classname;
            final RuleRefiner ruleRefiner = (RuleRefiner) createByClassname(fullyQualifiedClassName);
            ruleRefiner.setProperties(properties);
            ruleRefiner.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setRuleRefiner(ruleRefiner);
        } else if (nodeName.equals("rulestoppingcriterion") || nodeName.equals("rsc")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".rulestoppingcriterions." + classname;
            final RuleStoppingCriterion ruleStoppingCriterion = (RuleStoppingCriterion) createByClassname(fullyQualifiedClassName);
            ruleStoppingCriterion.setProperties(properties);
            ruleStoppingCriterion.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setRuleStoppingCriterion(ruleStoppingCriterion);
        } else if (nodeName.equals("stoppingcriterion") || nodeName.equals("sc")) {
            final String fullyQualifiedClassName = COMPONENTS_PATH + ".stoppingcriterions." + classname;
            final StoppingCriterion stoppingCriterion = (StoppingCriterion) createByClassname(fullyQualifiedClassName);
            stoppingCriterion.setProperties(properties);
            stoppingCriterion.setRandom(seCoAlgorithm.getRandom());
            seCoAlgorithm.setStoppingCriterion(stoppingCriterion);
        }
    }

    private static Hashtable<String, String> getProperties(final XmlPullParser xmlPullParser) {
        final Hashtable<String, String> attributes = new Hashtable<String, String>();

        for (int attributeIndex = 0; attributeIndex < xmlPullParser.getAttributeCount(); attributeIndex++)
            attributes.put(xmlPullParser.getAttributeName(attributeIndex), xmlPullParser.getAttributeValue(attributeIndex));

        return attributes;
    }

    private static Object createByClassname(final String classname) {
        try {
            return Class.forName(classname).getConstructor(new Class[0]).newInstance(new Object[0]);
        } catch (final Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
