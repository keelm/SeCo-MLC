package de.tu_darmstadt.ke.seco.utils;

import java.io.File;

import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * The configuration for the entire framework is accessed by this class.
 * <p>
 * The {@link FrameworkConfig} reads a XML file to access the desired configuration of the user. Once the class is initialized the configuration can be accessed by {@link FrameworkConfig#getConfiguration(String, String, Class)}.
 * <p>
 * 
 * @author Markus Zopf
 * 
 */

public final class FrameworkConfig {

	/**
	 * The path where the framework configuration file is located.
	 */
	private static final String FRAMEWORK_CONFIG_PATH = "config/";

	/**
	 * The framework configuration file name.
	 */
	private static final String FRAMEWORK_CONFIG_FILENAME = "FrameworkConfig.xml";

	/**
	 * The instance of the framework configuration according to the singleton pattern.
	 */
	private static FrameworkConfig instance;

	/**
	 * The configuration document, which is generated out of the configuration file.
	 */
	private Document configDocument = null;

	/**
	 * Private constructor according to the singleton pattern.
	 * 
	 * @param configPath
	 *            the path where the configuration file is located
	 * @param configFileName
	 *            the name of the configuration file
	 */
	private FrameworkConfig(final String configPath, final String configFileName) {
		final File configFile = new File(configPath + configFileName);

		try {
			configDocument = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(configFile);
			configDocument.getDocumentElement().normalize();
		}
		catch (final Exception e) {
			Logger.warn("Could not read config file '" + configPath + configFileName + ". Using default configuration.", e);
		}
	}

	/**
	 * Private access to the logger instance according to the singleton pattern.
	 * 
	 * @return the framework configuration instance
	 */
	private static FrameworkConfig getInstance() {
		if (instance == null)
			instance = new FrameworkConfig(FRAMEWORK_CONFIG_PATH, FRAMEWORK_CONFIG_FILENAME);

		return instance;
	}

	/**
	 * Gets the configuration entry for the specific tag in the specific group.
	 * 
	 * @param <T>
	 *            the type/class of the desired configuration element
	 * @param configurationGroup
	 *            the group for the <code>configurationTag</code>
	 * @param configurationTag
	 *            the tag of the configuration element
	 * @param c
	 *            the type/class of the desired configuration element
	 * @return the desired configuration element
	 */
	@SuppressWarnings("unchecked")
	public static <T> T getConfiguration(final String configurationGroup, final String configurationTag, final Class<T> c) {
		if (getInstance().configDocument == null)
			return null;

		final NodeList configurationTagEntries = getInstance().configDocument.getElementsByTagName(configurationGroup);

		if (configurationTagEntries.getLength() == 0) {
			Logger.warn("No configuration entry for group '" + configurationGroup + "' found.");
			return null;
		}

		if (configurationTagEntries.getLength() > 1) {
			Logger.warn("More than one configuration entry for group '" + configurationGroup + "' found.");
			return null;
		}

		T configurationObject = null;
		boolean configurationWasInvalid = false;

		final NodeList configurationEntries = configurationTagEntries.item(0).getChildNodes();
		for (int entryIndex = 0; entryIndex < configurationEntries.getLength(); entryIndex++) {
			final Node configurationEntry = configurationEntries.item(entryIndex);
			if (configurationEntry.getNodeType() == Node.ELEMENT_NODE && configurationEntry.getNodeName().equals(configurationTag))
				try {
					if (configurationObject != null)
						Logger.warn("Value for configuration tag '" + configurationTag + "' in configuration group '" + configurationGroup + "' already configured. Overwriting previous configuration.");

					configurationObject = (T) TextToClassParser.parse(configurationEntry.getTextContent(), c);
					if (configurationObject == null)
						throw new NullPointerException();
				}
				catch (final Exception e) {
					Logger.warn("Value '" + configurationEntry.getTextContent() + "' for configuration tag '" + configurationTag + "' in configuration group '" + configurationGroup + "' not valid.");
					configurationWasInvalid = true;
				}
		}

		if (configurationObject == null && !configurationWasInvalid)
			Logger.debug("Configuration tag '" + configurationTag + "' in configuration group '" + configurationGroup + "' not found.");

		return configurationObject;
	}
}
