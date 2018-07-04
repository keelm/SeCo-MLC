package de.tu_darmstadt.ke.seco.utils;

import java.lang.reflect.Method;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;

/**
 * Simple, lightweight logger to log system messages.
 * 
 * @author Markus Zopf
 */
public final class Logger {

	static {
		// TODO Maybe put this initialization to another spot. When there is a well defined starting point of the framework, this would be a good place for this lines.
		Logger.info("Initialising the framework.");
		Logger.initByFrameworkConfig();
	}

	/**
	 * The log levels which the logger can distinguish.
	 * 
	 * @author Markus Zopf
	 * 
	 */
	public enum LogLevel {
		/**
		 * Should be used to log serious errors that influence the behavior of the system dramatically.
		 */
		ERROR,

		/**
		 * Should be used to log warnings that describe abnormal system behaviors.
		 */
		WARN,

		/**
		 * Should be used to log info messages that occur during normal system behavior.
		 */
		INFO,

		/**
		 * Should be used only to log messages that give a deeper insight to the system functions and are not of interest to a normal user.
		 */
		DEBUG;

		/**
		 * Parses the string argument to a {@link LogLevel}.
		 * 
		 * @param logLevel
		 *            the string to parse
		 * @return the given string as a {@link LogLevel}
		 * @throws IllegalArgumentException
		 *             if the given string is invalid
		 * @throws NullPointerException
		 *             if the given string is {@code null}
		 */
		public static LogLevel parseLogLevel(final String logLevel) throws IllegalArgumentException, NullPointerException {
			if (logLevel == null)
				throw new NullPointerException();
			else if (logLevel.equalsIgnoreCase("error"))
				return ERROR;
			else if (logLevel.equalsIgnoreCase("warn"))
				return WARN;
			else if (logLevel.equalsIgnoreCase("info"))
				return INFO;
			else if (logLevel.equalsIgnoreCase("debug"))
				return DEBUG;
			else
				throw new IllegalArgumentException();
		}
	}

	/**
	 * The search depth for evaluating the class which is calling a function.
	 */
	private static final int CALLING_CLASS_SEARCH_DEPTH = 4;

	/**
	 * The instance of the logger according to the singleton pattern.
	 */
	private static Logger instance;

	/**
	 * The date format which is used to format the date in the logger outputs.
	 */
	private SimpleDateFormat simpleDateFormat = new SimpleDateFormat("HH:mm:ss");;

	/**
	 * To enable or disable the logging. Messages are only logged if the logging is enabled.
	 */
	private boolean enabled = true;

	/**
	 * Show exceptions at all log levels or just the exceptions which occurred with log level {@link LogLevel#ERROR}.
	 */
	private boolean showAllExceptions = false;

	/**
	 * The maximum {@link LogLevel} to log.
	 * <p>
	 * E.g. {@code logLevel = LogLevel.WARN} will log only {@link LogLevel#ERROR} and {@link LogLevel#WARN} messages.
	 */
	private LogLevel logLevel = LogLevel.ERROR;

	/**
	 * Private constructor according to the singleton pattern.
	 */
	private Logger() {

	}

	/**
	 * Initialize the Logger with data given by the {@link FrameworkConfig}.
	 */
	public static void initByFrameworkConfig() {
		final Boolean newEnabled = FrameworkConfig.getConfiguration("Logger", "Enabled", Boolean.class);
		if (newEnabled != null)
			getInstance().enabled = newEnabled;

		final LogLevel newLogLevel = FrameworkConfig.getConfiguration("Logger", "LogLevel", LogLevel.class);
		if (newLogLevel != null)
			getInstance().logLevel = newLogLevel;

		final SimpleDateFormat newSimpleDateFormat = FrameworkConfig.getConfiguration("Logger", "DateFormat", SimpleDateFormat.class);
		if (newSimpleDateFormat != null)
			getInstance().simpleDateFormat = newSimpleDateFormat;

		final Boolean newShowAllExceptions = FrameworkConfig.getConfiguration("Logger", "ShowAllExceptions", Boolean.class);
		if (newShowAllExceptions != null)
			getInstance().showAllExceptions = newShowAllExceptions;
	}

	/**
	 * Private access to the logger instance according to the singleton pattern.
	 * 
	 * @return the logger instance
	 */
	private static Logger getInstance() {
		if (instance == null)
			instance = new Logger();

		return instance;
	}

	/**
	 * Logs depending on the {@link #logLevel} and the {@link #enabled} status a message.
	 * 
	 * @param logLevelForMessage
	 *            the {@link LogLevel} the message belongs to
	 * @param messageToLog
	 *            the message to log
	 * @param throwable
	 *            the throwable which occurred
	 * @return true, if the message was logged; false otherwise
	 */
	private boolean log(final LogLevel logLevelForMessage, final String messageToLog, final Throwable throwable) {
		if (enabled && logLevel.compareTo(logLevelForMessage) >= 0) {
			final StringBuilder logMessageBuilder = new StringBuilder();
			logMessageBuilder.append("[");
			logMessageBuilder.append(simpleDateFormat.format(new Date()));
			logMessageBuilder.append(" | ");
			logMessageBuilder.append(logLevelForMessage);
			logMessageBuilder.append(" | ");
			logMessageBuilder.append(getCallingClassName(CALLING_CLASS_SEARCH_DEPTH));
			logMessageBuilder.append("]: ");

			final String indentString = generateStringWithLength(logMessageBuilder.length(), ' ');
			logMessageBuilder.append(messageToLog.replaceAll("\\n", "\n" + indentString));

			if (logLevelForMessage == LogLevel.ERROR) {
				System.err.println(logMessageBuilder.toString());
				if (throwable != null)
					throwable.printStackTrace(System.err);
			}
			else if (logLevelForMessage == LogLevel.WARN) {
				System.err.println(logMessageBuilder.toString());
				if (throwable != null && showAllExceptions)
					throwable.printStackTrace(System.err);
			}
			else {
				System.out.println(logMessageBuilder.toString());
				if (throwable != null && showAllExceptions)
					throwable.printStackTrace(System.out);
			}

			return true;
		}

		else
			return false;
	}

	/**
	 * Logs a message with log level {@link LogLevel#ERROR}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean error(final String messageToLog) {
		return Logger.getInstance().log(LogLevel.ERROR, messageToLog, null);
	}

	/**
	 * Logs a message with log level {@link LogLevel#ERROR} and an occurred {@link Throwable}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @param throwable
	 *            the occurred {@link Throwable}
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean error(final String messageToLog, final Throwable throwable) {
		return Logger.getInstance().log(LogLevel.ERROR, messageToLog, throwable);
	}

	/**
	 * Logs a message with log level {@link LogLevel#WARN}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean warn(final String messageToLog) {
		return Logger.getInstance().log(LogLevel.WARN, messageToLog, null);
	}

	/**
	 * Logs a message with log level {@link LogLevel#WARN} and an occurred {@link Throwable}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @param throwable
	 *            the occurred {@link Throwable}
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean warn(final String messageToLog, final Throwable throwable) {
		return Logger.getInstance().log(LogLevel.WARN, messageToLog, throwable);
	}

	/**
	 * Logs a message with log level {@link LogLevel#INFO}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean info(final String messageToLog) {
		return Logger.getInstance().log(LogLevel.INFO, messageToLog, null);
	}

	/**
	 * Logs a message with log level {@link LogLevel#INFO} and an occurred {@link Throwable}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @param throwable
	 *            the occurred {@link Throwable}
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean info(final String messageToLog, final Throwable throwable) {
		return Logger.getInstance().log(LogLevel.INFO, messageToLog, throwable);
	}

	/**
	 * Logs a message with log level {@link LogLevel#DEBUG}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean debug(final String messageToLog) {
		return Logger.getInstance().log(LogLevel.DEBUG, messageToLog, null);
	}

	/**
	 * Logs a message with log level {@link LogLevel#DEBUG} and an occurred {@link Throwable}.
	 * 
	 * @param messageToLog
	 *            the message to log
	 * @param throwable
	 *            the occurred {@link Throwable}
	 * @return true, if the message was logged; false otherwise
	 */
	public static boolean debug(final String messageToLog, final Throwable throwable) {
		return Logger.getInstance().log(LogLevel.DEBUG, messageToLog, throwable);
	}

	/**
	 * Enables the logging (sets {@link Logger#enabled} to {@code true}).
	 */
	public static void enableLogging() {
		Logger.getInstance().enabled = true;
	}

	/**
	 * Disables the logging (sets {@link Logger#enabled} to {@code false}).
	 */
	public static void disableLogging() {
		Logger.getInstance().enabled = false;
	}

	/**
	 * Sets {@link Logger#logLevel}.
	 * 
	 * @param newLogLevel
	 *            the new log level
	 */
	public static void setLogLevel(final LogLevel newLogLevel) {
		Logger.getInstance().logLevel = newLogLevel;
	}

	/**
	 * Returns {@link Logger#logLevel}.
	 * 
	 * @return the current{@link LogLevel} of the logger
	 */
	public static LogLevel getLogLevel() {
		return Logger.getInstance().logLevel;
	}

	/**
	 * Returns the calling class using reflection.
	 * 
	 * @param level
	 *            how deep the function should look for the calling class
	 * @return the class which called the function
	 */
	private static String getCallingClassName(final int level) {
		String callingClassName;
		try {
			final Class<?> sunReflectionClass = Class.forName("sun.reflect.Reflection");
			final Object sunReflection = sun.reflect.Reflection.class.newInstance();
			final Method sunReflectionGetCallerClassMethod = sunReflectionClass.getMethod("getCallerClass", new Class[] { Integer.TYPE });

			callingClassName = ((Class<?>) sunReflectionGetCallerClassMethod.invoke(sunReflection, new Object[] { level })).getName();
			callingClassName = callingClassName.substring(callingClassName.lastIndexOf('.') + 1);
		}

		catch (final Exception e) {
			callingClassName = "";
			e.printStackTrace();
		}
		return callingClassName;
	}

	/**
	 * Generates a {@link String} of length {@code length} filled with the character {@code fillCharacter}.
	 * 
	 * @param length
	 *            the length of the generated string
	 * @param fillCharacter
	 *            the character the string is filled with
	 * @return a string of length {@code length filled with the character {@code fillCharacter}
	 */
	private String generateStringWithLength(final int length, final char fillCharacter) {
		if (length < 0)
			return null;
		final char[] array = new char[length];
		Arrays.fill(array, fillCharacter);
		return new String(array);
	}
}
