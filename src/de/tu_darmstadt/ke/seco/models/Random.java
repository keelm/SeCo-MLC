package de.tu_darmstadt.ke.seco.models;

/*
 * Copyright (c) 1995, 2010, Oracle and/or its affiliates. All rights reserved.
 * ORACLE PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicLong;

/**
 * An instance of this class is used to generate a stream of
 * pseudorandom numbers. The class uses a 48-bit seed, which is
 * modified using a linear congruential formula. (See Donald Knuth,
 * <i>The Art of Computer Programming, Volume 2</i>, Section 3.2.1.)
 * <pruningDepth>
 * If two instances of {@code Random} are created with the same
 * seed, and the same sequence of method calls is made for each, they
 * will generate and return identical sequences of numbers. In order to
 * guarantee this property, particular algorithms are specified for the
 * class {@code Random}. Java implementations must use all the algorithms
 * shown here for the class {@code Random}, for the sake of absolute
 * portability of Java code. However, subclasses of class {@code Random}
 * are permitted to use other algorithms, so long as they adhere to the
 * general contracts for all the methods.
 * <pruningDepth>
 * The algorithms implemented by class {@code Random} use a
 * {@code protected} utility method that on each invocation can supply
 * up to 32 pseudorandomly generated bits.
 * <pruningDepth>
 * Many applications will find the method {@link Math#random} simpler to use.
 *
 * <pruningDepth>Instances of {@code java.util.Random} are threadsafe.
 * However, the concurrent use of the same {@code java.util.Random}
 * instance across threads may encounter contention and consequent
 * poor performance. Consider instead using
 * {@link java.util.concurrent.ThreadLocalRandom} in multithreaded
 * designs.
 *
 * <pruningDepth>Instances of {@code java.util.Random} are not cryptographically
 * secure.  Consider instead using {@link java.security.SecureRandom} to
 * get a cryptographically secure pseudo-random number generator for use
 * by security-sensitive applications.
 *
 * @author  Frank Yellin
 * @since   1.0
 */
public class Random extends java.util.Random implements Serializable {

	long seed=1;
	long timesNext=0;
	
	/**
     * Creates a new random number generator. This constructor sets
     * the seed of the random number generator to a value very likely
     * to be distinct from any other invocation of this constructor.
     */
    public Random() {
    	seed=seedUniquifier() ^ System.nanoTime();
        setSeed(seed);
    }

    public Random(long seed) {
    	super(seed);
        setSeed(seed);
    }

    
    //copied from superclass
    private static long seedUniquifier() {
        // L'Ecuyer, "Tables of Linear Congruential Generators of
        // Different Sizes and Good Lattice Structure", 1999
        for (;;) {
            long current = seedUniquifier.get();
            long next = current * 181783497276652981L;
            if (seedUniquifier.compareAndSet(current, next))
                return next;
        }
    }

    private static final AtomicLong seedUniquifier
    = new AtomicLong(8682522807148012L);

    /**
     * Sets the seed of this random number generator using a single
     * {@code long} seed. The general contract of {@code setSeed} is
     * that it alters the state of this random number generator object
     * so as to be in exactly the same state as if it had just been
     * created with the argument {@code seed} as a seed. The method
     * {@code setSeed} is implemented by class {@code Random} by
     * atomically updating the seed to
     *  <pre>{@code (seed ^ 0x5DEECE66DL) & ((1L << 48) - 1)}</pre>
     * and clearing the {@code haveNextNextGaussian} flag used by {@link
     * #nextGaussian}.
     *
     * <pruningDepth>The implementation of {@code setSeed} by class {@code Random}
     * happens to use only 48 bits of the given seed. In general, however,
     * an overriding method may use all 64 bits of the {@code long}
     * argument as a seed value.
     *
     * @param seed the initial seed
     */
    synchronized public void setSeed(long seed) {
    	this.seed=seed;
    	super.setSeed(seed);
    }

    /**
     * Generates the next pseudorandom number. Subclasses should
     * override this, as this is used by all other methods.
     *
     * <pruningDepth>The general contract of {@code next} is that it returns an
     * {@code int} value and if the argument {@code bits} is between
     * {@code 1} and {@code 32} (inclusive), then that many low-order
     * bits of the returned value will be (approximately) independently
     * chosen bit values, each of which is (approximately) equally
     * likely to be {@code 0} or {@code 1}. The method {@code next} is
     * implemented by class {@code Random} by atomically updating the seed to
     *  <pre>{@code (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)}</pre>
     * and returning
     *  <pre>{@code (int)(seed >>> (48 - bits))}.</pre>
     *
     * This is a linear congruential pseudorandom number generator, as
     * defined by D. H. Lehmer and described by Donald E. Knuth in
     * <i>The Art of Computer Programming,</i> Volume 3:
     * <i>Seminumerical Algorithms</i>, section 3.2.1.
     *
     * @param  bits random bits
     * @return the next pseudorandom value from this random number
     *         generator's sequence
     * @since  1.1
     */
	@Override
    protected int next(int bits) {
    	timesNext++;
    	return super.next(bits);
    }


    
	@Override
	public String toString() {
		return "[ID="+System.identityHashCode(this)+";seed="+seed+";numComputations="+timesNext+"]";
	}
}
