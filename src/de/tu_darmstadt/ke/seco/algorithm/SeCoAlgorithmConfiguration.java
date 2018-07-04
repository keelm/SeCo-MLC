package de.tu_darmstadt.ke.seco.algorithm;

import java.io.Serializable;
import java.io.StringWriter;
import java.lang.reflect.Field;
import java.util.Hashtable;
import java.util.Map.Entry;
import java.util.Random;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors.CandidateSelector;
import de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors.SelectAllCandidatesSelector;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.MEstimate;
import de.tu_darmstadt.ke.seco.algorithm.components.postprocessors.NoOpPostProcessor;
import de.tu_darmstadt.ke.seco.algorithm.components.postprocessors.PostProcessor;
import de.tu_darmstadt.ke.seco.algorithm.components.rulefilters.BeamWidthFilter;
import de.tu_darmstadt.ke.seco.algorithm.components.rulefilters.RuleFilter;
import de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers.RuleInitializer;
import de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers.TopDownRuleInitializer;
import de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners.RuleRefiner;
import de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners.TopDownRefiner;
import de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions.CoverageRuleStop;
import de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions.RuleStoppingCriterion;
import de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions.NoNegativesCoveredStop;
import de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions.StoppingCriterion;

public class SeCoAlgorithmConfiguration implements Serializable {

    private static final long serialVersionUID = 2895086346619413864L;

    /**
     * the name of the classifier configuration
     */
    private String configurationName;

    /**
     * The percentage size of the growing set. I.e. a value of 1 means "use 100% for growing (no pruning)", a value of 0.8 means "use 80% for growing and 20% for pruning"
     */
    private double growingSetSize = 1;

    /**
     * The minimum number of examples a rule has to cover
     * <pruningDepth>
     * TODO by m.zopf: should very likely be placed in another class (maybe in the RuleFilter)
     */
    private int minNo = 1;

    /**
     * used for stratifying the folds of a cross validation in a random way
     */
    private static Random random = new Random(); // generate a default random object

    private boolean strictlyGreater = false;

    private CandidateSelector candidateSelector;
    private Heuristic heuristic;
    private PostProcessor postProcessor;
    private RuleFilter ruleFilter;
    private RuleInitializer ruleInitializer;
    private RuleRefiner ruleRefiner;
    private RuleStoppingCriterion ruleStoppingCriterion;
    private StoppingCriterion stoppingCriterion;

    public SeCoAlgorithmConfiguration(String configurationName) {
        this.configurationName = configurationName;
        initDefaultConfiguration();
    }

    private void initDefaultConfiguration() {
        candidateSelector = new SelectAllCandidatesSelector();
        heuristic = new MEstimate();
        postProcessor = new NoOpPostProcessor();
        ruleFilter = new BeamWidthFilter();
        ruleInitializer = new TopDownRuleInitializer();
        ruleRefiner = new TopDownRefiner();
        ruleStoppingCriterion = new CoverageRuleStop();
        stoppingCriterion = new NoNegativesCoveredStop();
    }

    public void setProperty(final String name, final String value) {
        if (name.equalsIgnoreCase("growingSetSize")) {
            double size;
            if (value.contains("/")) {
                final String[] fractionParts = value.replaceAll(" ", "").split("/");
                if (fractionParts.length != 2) {
                    throw new NumberFormatException("Could not parse growingSetSize. growingSetSize was '" + value + "'.");
                } else {
                    size = Double.parseDouble(fractionParts[0]) / Double.parseDouble(fractionParts[1]);
                }
            } else {
                size = Double.parseDouble(value);
            }

            if (size > 1) {
                throw new IllegalArgumentException("growingSetSize must be <= 1. growingSetSize was '" + size + "'.");
            } else {
                growingSetSize = size;
            }
        } else if (name.equalsIgnoreCase("minNo")) {
            Integer size = Integer.parseInt(value);
            if (size < 1) {
                size = 1;
            }
            minNo = size;
        } else if (name.equalsIgnoreCase("seed")) {
            random = new Random(Long.parseLong(value));
        } else if (name.equalsIgnoreCase("strictlyGreater")) {
            strictlyGreater = Boolean.parseBoolean(value);
        }
    }

    public void setAttributes(final Hashtable<String, String> attributes) {
        for (final Entry<String, String> attribute : attributes.entrySet()) {
            setProperty(attribute.getKey(), attribute.getValue());
        }
    }

    public void setCandidateSelector(final CandidateSelector candidateSelector) {
        this.candidateSelector = candidateSelector;
    }

    public void setHeuristic(final Heuristic heuristic) {
        this.heuristic = heuristic;
    }

    public void setPostProcessor(final PostProcessor postProcessor) {
        this.postProcessor = postProcessor;
    }

    public void setRuleFilter(final RuleFilter ruleFilter) {
        this.ruleFilter = ruleFilter;
    }

    public void setRuleInitializer(final RuleInitializer ruleInitializer) {
        this.ruleInitializer = ruleInitializer;
    }

    public void setRuleRefiner(final RuleRefiner ruleRefiner) {
        this.ruleRefiner = ruleRefiner;
    }

    public void setRuleStoppingCriterion(final RuleStoppingCriterion ruleStoppingCriterion) {
        this.ruleStoppingCriterion = ruleStoppingCriterion;
    }

    public void setStoppingCriterion(final StoppingCriterion stoppingCriterion) {
        this.stoppingCriterion = stoppingCriterion;
    }

    public CandidateSelector getCandidateSelector() {
        return candidateSelector;
    }

    public Heuristic getHeuristic() {
        return heuristic;
    }

    public PostProcessor getPostProcessor() {
        return postProcessor;
    }

    public RuleFilter getRuleFilter() {
        return ruleFilter;
    }

    public RuleInitializer getRuleInitializer() {
        return ruleInitializer;
    }

    public RuleRefiner getRuleRefiner() {
        return ruleRefiner;
    }

    public RuleStoppingCriterion getRuleStoppingCriterion() {
        return ruleStoppingCriterion;
    }

    public StoppingCriterion getStoppingCriterion() {
        return stoppingCriterion;
    }

    @Override
    public String toString() {
        final StringBuilder stringBuilder = new StringBuilder();

        String string = (new ReflectionToStringBuilder(this, ToStringStyle.SHORT_PREFIX_STYLE) {
            @Override
            protected boolean accept(final Field f) {
                return super.accept(f) && !(f.getName().equals("candidateSelector") || f.getName().equals("heuristic") || f.getName().equals("postProcessor") || f.getName().equals("ruleFilter") || f.getName().equals("ruleInitializer") || f.getName().equals("ruleRefiner") || f.getName().equals("ruleStoppingCriterion") || f.getName().equals("stoppingCriterion") || f.getName().equals("weightModel"));
            }
        }).toString();

        string = string.replaceAll("\\[\\]", "");
        stringBuilder.append("seCoAlgorithm........: " + string + "\n");

        stringBuilder.append("candidateSelector....: " + candidateSelector + "\n");
        stringBuilder.append("heuristic............: " + heuristic + "\n");
        stringBuilder.append("postProcessor........: " + postProcessor + "\n");
        stringBuilder.append("ruleFilter...........: " + ruleFilter + "\n");
        stringBuilder.append("ruleInitializer......: " + ruleInitializer + "\n");
        stringBuilder.append("ruleRefiner..........: " + ruleRefiner + "\n");
        stringBuilder.append("ruleStoppingCriterion: " + ruleStoppingCriterion + "\n");
        stringBuilder.append("stoppingCriterion....: " + stoppingCriterion + "\n");
        return stringBuilder.toString();
    }

    public int getMinNo() {
        return minNo;
    }

    public void setMinNo(final int minNo) {
        this.minNo = minNo;
    }

    public static Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        SeCoAlgorithmConfiguration.random = random;
    }

    public double getGrowingSetSize() {
        return growingSetSize;
    }

    public void setGrowingSetSize(final double growingSetSize) {
        this.growingSetSize = growingSetSize;
    }

    public String getName() {
        return configurationName;
    }

    public boolean isStrictlyGreater() {
        return strictlyGreater;
    }

    public String toXMLConfigurationString() throws Exception {
        Document document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
        Element seco = document.createElement("seco");
        seco.setAttribute("growingSetSize", String.valueOf(growingSetSize));
        seco.setAttribute("minNo", String.valueOf(minNo));
        seco.setAttribute("strictlyGreater", String.valueOf(strictlyGreater));
        // seco.setAttribute("seed", "0"); // TODO by m.zopf: how could the seed be set? it must be configurable in the GUI (set/not set)
        document.appendChild(seco);

		/*
        seco.appendChild(candidateSelector.toXMLConfigurationString(document));
		seco.appendChild(heuristic.toXMLConfigurationString(document));
		seco.appendChild(postProcessor.toXMLConfigurationString(document));
		seco.appendChild(ruleFilter.toXMLConfigurationString(document));
		seco.appendChild(ruleInitializer.toXMLConfigurationString(document));
		seco.appendChild(ruleRefiner.toXMLConfigurationString(document));
		seco.appendChild(ruleStoppingCriterion.toXMLConfigurationString(document));
		seco.appendChild(stoppingCriterion.toXMLConfigurationString(document));
		*/

        return getStringFromDocument(document);
    }

    private static String getStringFromDocument(Document doc) throws TransformerException {
        StringWriter sw = new StringWriter();
        TransformerFactory tf = TransformerFactory.newInstance();
        Transformer transformer = tf.newTransformer();
        transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty(OutputKeys.ENCODING, "UTF-8");

        transformer.transform(new DOMSource(doc), new StreamResult(sw));
        return sw.toString();
    }
}
