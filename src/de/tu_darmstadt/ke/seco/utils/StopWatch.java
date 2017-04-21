package de.tu_darmstadt.ke.seco.utils;

public class StopWatch {

	private long startTime = 0;
	private long stopTime = 0;
	private boolean running = false;

	public void start() {
		this.startTime = System.currentTimeMillis();
		this.running = true;
	}

	public void stop() {
		this.stopTime = System.currentTimeMillis();
		this.running = false;
	}

	public void reset() {
		startTime = 0;
		stopTime = 0;
		running = false;
	}

	public long getElapsedTime() {
		if (running)
			return (System.currentTimeMillis() - startTime);
		else
			return stopTime - startTime;
	}

	public double getElapsedTimeSec() {
		return getElapsedTime() / 1000.0;
	}

}