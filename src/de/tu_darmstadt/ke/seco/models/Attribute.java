package de.tu_darmstadt.ke.seco.models;

import weka.core.SerializedObject;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;

public class Attribute extends weka.core.Attribute implements Comparable<Attribute> {
	public Attribute(String attributeName) {
		super(attributeName);
	}

	
	
	public Attribute(String attributeName, List<String> attributeValues) {
		super(attributeName, attributeValues);
	}

	
	private static HashMap<weka.core.Attribute, Attribute> wekaToSeCoAttributes = new HashMap<>();

	
	public static Attribute toSeCoAttribute(weka.core.Attribute wekaAttribute) {
		Attribute wekaAttributeFromCache = wekaToSeCoAttributes.get(wekaAttribute);

		if (wekaAttributeFromCache != null)
			return wekaAttributeFromCache;

		else {
			Attribute seCoAttribute = new Attribute(wekaAttribute.name());

			try {
				Field m_IndexField = weka.core.Attribute.class.getDeclaredField("m_Index");
				Field m_TypeField = weka.core.Attribute.class.getDeclaredField("m_Type");
				Field m_ValuesField = weka.core.Attribute.class.getDeclaredField("m_Values");
				Field m_HashtableField = weka.core.Attribute.class.getDeclaredField("m_Hashtable");
				Field m_DateFormatField = weka.core.Attribute.class.getDeclaredField("m_DateFormat");
				Field m_Header = weka.core.Attribute.class.getDeclaredField("m_Header");
				Field m_Metadata = weka.core.Attribute.class.getDeclaredField("m_Metadata");

				m_IndexField.setAccessible(true);
				m_TypeField.setAccessible(true);
				m_ValuesField.setAccessible(true);
				m_HashtableField.setAccessible(true);
				m_DateFormatField.setAccessible(true);
				m_Header.setAccessible(true);
				m_Metadata.setAccessible(true);

				m_IndexField.set(seCoAttribute, m_IndexField.get(wekaAttribute));
				m_TypeField.set(seCoAttribute, m_TypeField.get(wekaAttribute));
				m_ValuesField.set(seCoAttribute, m_ValuesField.get(wekaAttribute));
				m_HashtableField.set(seCoAttribute, m_HashtableField.get(wekaAttribute));
				m_DateFormatField.set(seCoAttribute, m_DateFormatField.get(wekaAttribute));
				m_Header.set(seCoAttribute, m_Header.get(wekaAttribute));
				m_Metadata.set(seCoAttribute, m_Metadata.get(wekaAttribute));
			}

			catch (NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException e) {
				e.printStackTrace();
			}

			wekaToSeCoAttributes.put(wekaAttribute, seCoAttribute);
			return seCoAttribute;
		}
	}

	@Override
	public int compareTo(Attribute o) {
		return this.index() - o.index();
	}

	final void setValue(int index, String string) {
		try {
			Field m_ValuesField = weka.core.Attribute.class.getDeclaredField("m_Values");
			Field m_HashtableField = weka.core.Attribute.class.getDeclaredField("m_Hashtable");
			m_ValuesField.setAccessible(true);
			m_HashtableField.setAccessible(true);

			switch (type()) {
			case NOMINAL:
			case STRING:
				m_ValuesField.set(this, m_ValuesField.get(this));
				m_HashtableField.set(this, m_HashtableField.get(this));
				Object store = string;
				if (string.length() > 200)
					try {
						store = new SerializedObject(string, true);
					}
					catch (Exception ex) {
						System.err.println("Couldn't compress string attribute value -" + " storing uncompressed.");
					}
				((Hashtable<Object, Integer>) m_HashtableField.get(this)).remove(((ArrayList<Object>) m_ValuesField.get(this)).get(index));
				((ArrayList<Object>) m_ValuesField.get(this)).set(index, store);
				((Hashtable<Object, Integer>) m_HashtableField.get(this)).put(store, new Integer(index));
				break;
			default:
				throw new IllegalArgumentException("Can only set values for nominal" + " or string attributes!");
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}

}
