package de.tu_darmstadt.ke.seco.algorithm;

import java.io.File;
import java.util.Hashtable;

import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

import de.tu_darmstadt.ke.seco.algorithm.components.candidateselectors.CandidateSelector;
import de.tu_darmstadt.ke.seco.algorithm.components.heuristics.Heuristic;
import de.tu_darmstadt.ke.seco.algorithm.components.postprocessors.PostProcessor;
import de.tu_darmstadt.ke.seco.algorithm.components.rulefilters.RuleFilter;
import de.tu_darmstadt.ke.seco.algorithm.components.ruleinitializers.RuleInitializer;
import de.tu_darmstadt.ke.seco.algorithm.components.rulerefiners.RuleRefiner;
import de.tu_darmstadt.ke.seco.algorithm.components.rulestoppingcriterions.RuleStoppingCriterion;
import de.tu_darmstadt.ke.seco.algorithm.components.stoppingcriterions.StoppingCriterion;

public class SeCoAlgorithmConfigurationFactory {

	public static String COMPONENTS_PATH = "de.tu_darmstadt.ke.seco.algorithm.components";
	public static String CANDIDATE_SELECTORS_PATH = COMPONENTS_PATH + ".candidateselectors";
	public static String HEURISTICS_PATH = COMPONENTS_PATH + ".heuristics";
	public static String POST_PROCESSORS_PATH = COMPONENTS_PATH + ".postprocessors";
	public static String RULE_FILTERS_PATH = COMPONENTS_PATH + ".rulefilters";
	public static String RULE_INITIALIZERS_PATH = COMPONENTS_PATH + ".ruleinitializers";
	public static String RULE_REFINERS_PATH = COMPONENTS_PATH + ".rulerefiners";
	public static String RULE_STOPPING_CRITERIONS_PATH = COMPONENTS_PATH + ".rulestoppingcriterions";
	public static String STOPPING_CRITERIONS_PATH = COMPONENTS_PATH + ".stoppingcriterions";

	public static SeCoAlgorithmConfiguration buildAlgorithmFromFile(final String filename) throws Exception {
		return buildAlgorithmFromFile(new File(filename));
	}

	public static SeCoAlgorithmConfiguration buildAlgorithmFromFile(final File file) throws Exception {
		Document document = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file);
		return buildAlgorithmFromDOMDocument(file.getName(), document);
	}

	private static SeCoAlgorithmConfiguration buildAlgorithmFromDOMDocument(final String filename, Document document) throws Exception {
		final SeCoAlgorithmConfiguration seCoAlgorithmConfiguration = new SeCoAlgorithmConfiguration(filename);

		Element element = document.getDocumentElement();

		if (element.getNodeName().equals("seco")) {
			seCoAlgorithmConfiguration.setAttributes(getProperties(element.getAttributes()));
		}

		else {
			throw new Exception("XML configuration file has to start with seco node.");
		}

		for (int childIndex = 0; childIndex < element.getChildNodes().getLength(); childIndex++) {
			processSeCoCompenentNode(element.getChildNodes().item(childIndex), seCoAlgorithmConfiguration);
		}

		return seCoAlgorithmConfiguration;
	}

	private static void processSeCoCompenentNode(final Node childNode, final SeCoAlgorithmConfiguration seCoAlgorithmConfiguration) throws Exception {
		final String nodeName = childNode.getNodeName();

		final Hashtable<String, String> properties = getProperties(childNode.getAttributes());
		final String classname = properties.get("classname");
		properties.remove("classname");

		initializeComponent(seCoAlgorithmConfiguration, nodeName, classname, properties);
	}

	public static void initializeComponent(final SeCoAlgorithmConfiguration seCoAlgorithmConfiguration, final String componentName, final String componentClassname, final Hashtable<String, String> componentProperties) {
		if (componentName.equals("candidateselector") || componentName.equals("cs")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".candidateselectors." + componentClassname;
			final CandidateSelector candidateSelector = (CandidateSelector) createByClassname(fullyQualifiedClassName);
			candidateSelector.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setCandidateSelector(candidateSelector);
		}

		else if (componentName.equals("heuristic") || componentName.equals("h")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".heuristics." + componentClassname;
			final Heuristic heuristic = (Heuristic) createByClassname(fullyQualifiedClassName);
			heuristic.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setHeuristic(heuristic);
		}

		else if (componentName.equals("postprocessor") || componentName.equals("pp")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".postprocessors." + componentClassname;
			final PostProcessor postProcessor = (PostProcessor) createByClassname(fullyQualifiedClassName);
			postProcessor.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setPostProcessor(postProcessor);
		}

		else if (componentName.equals("rulefilter") || componentName.equals("rf")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".rulefilters." + componentClassname;
			final RuleFilter ruleFilter = (RuleFilter) createByClassname(fullyQualifiedClassName);
			ruleFilter.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setRuleFilter(ruleFilter);
		}

		else if (componentName.equals("ruleinitializer") || componentName.equals("ri")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".ruleinitializers." + componentClassname;
			final RuleInitializer ruleInitializer = (RuleInitializer) createByClassname(fullyQualifiedClassName);
			ruleInitializer.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setRuleInitializer(ruleInitializer);
		}

		else if (componentName.equals("rulerefiner") || componentName.equals("rr")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".rulerefiners." + componentClassname;
			final RuleRefiner ruleRefiner = (RuleRefiner) createByClassname(fullyQualifiedClassName);
			ruleRefiner.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setRuleRefiner(ruleRefiner);
		}

		else if (componentName.equals("rulestoppingcriterion") || componentName.equals("rsc")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".rulestoppingcriterions." + componentClassname;
			final RuleStoppingCriterion ruleStoppingCriterion = (RuleStoppingCriterion) createByClassname(fullyQualifiedClassName);
			ruleStoppingCriterion.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setRuleStoppingCriterion(ruleStoppingCriterion);
		}

		else if (componentName.equals("stoppingcriterion") || componentName.equals("sc")) {
			final String fullyQualifiedClassName = COMPONENTS_PATH + ".stoppingcriterions." + componentClassname;
			final StoppingCriterion stoppingCriterion = (StoppingCriterion) createByClassname(fullyQualifiedClassName);
			stoppingCriterion.setProperties(componentProperties);
			seCoAlgorithmConfiguration.setStoppingCriterion(stoppingCriterion);
		}
	}

	private static Hashtable<String, String> getProperties(NamedNodeMap namedNodeMap) {
		final Hashtable<String, String> attributes = new Hashtable<String, String>();

		if (namedNodeMap != null) {
			for (int attributeIndex = 0; attributeIndex < namedNodeMap.getLength(); attributeIndex++) {
				attributes.put(namedNodeMap.item(attributeIndex).getNodeName(), namedNodeMap.item(attributeIndex).getNodeValue());
			}
		}

		return attributes;
	}

	public static Object createByClassname(final String classname) {
		try {
			return Class.forName(classname).getConstructor(new Class[0]).newInstance(new Object[0]);
		}
		catch (final Exception e) {
			e.printStackTrace();
			return null;
		}
	}
}
