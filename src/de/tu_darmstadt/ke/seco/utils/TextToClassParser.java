package de.tu_darmstadt.ke.seco.utils;

import java.text.SimpleDateFormat;

public class TextToClassParser {

	/**
	 * Parses a string to the corresponding object of class <code>c</code>.
	 * 
	 * @param string
	 *            the string which should be parsed to a object
	 * @param c
	 *            the destination class of the object
	 * @return string <code>string</code> parsed to an object
	 */
	public static Object parse(String string, Class<?> c) {
		if (c.equals(Boolean.class))
			return Boolean.parseBoolean(string);

		else if (c.equals(SimpleDateFormat.class))
			return new SimpleDateFormat(string);

		else if (c.equals(Logger.LogLevel.class))
			return Logger.LogLevel.parseLogLevel(string);

		else
			throw new IllegalArgumentException("Parsing for class '" + c + "' not configured.");
	}
}
