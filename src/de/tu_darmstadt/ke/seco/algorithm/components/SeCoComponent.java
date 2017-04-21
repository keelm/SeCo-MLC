/*
 * This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * ISecoComponent.java Copyright (C) 2003-2010 Knowledge Engineering Group http://www.ke.tu-darmstadt.de
 *
 * Added by Matthias Thiel Created on 31.10.2004, 12:38
 */

package de.tu_darmstadt.ke.seco.algorithm.components;

import de.tu_darmstadt.ke.seco.models.Random;

import java.io.Serializable;
import java.util.Hashtable;
import java.util.Map.Entry;


import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

/**
 * A common interface for all components of a separate and conquer algorithm.
 *
 * @author Knowledge Engineering Group
 */
public abstract class SeCoComponent implements Serializable {

    private static final long serialVersionUID = 7469551269738858114L;
    protected Random random;

    /**
     * All components are able to receive configurable properties via this method that will overwrite the default values.
     *
     * @param name  the name of the property
     * @param value the value of the property
     */
    public void setProperty(final String name, final String value) {

    }

    public void setProperties(final Hashtable<String, String> properties) {
        for (final Entry<String, String> property : properties.entrySet())
            setProperty(property.getKey(), property.getValue());
    }

    public String getProperty(final String name) {
        return null;
    }

    public String getPropertiesString() {
        return "";
    }

    public void setRandom(Random random) {
        this.random = random;
    }

    @Override
    public String toString() {
        String string = ReflectionToStringBuilder.toString(this, ToStringStyle.SHORT_PREFIX_STYLE);
        return string.replaceAll("\\[\\]", "");
    }
}
