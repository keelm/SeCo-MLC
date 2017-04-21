package de.tu_darmstadt.ke.seco.utils;

public class TicksPerSecondCounter extends Thread {

	public static TicksPerSecondCounter globalTicksPerSecondCounter = new TicksPerSecondCounter();

	static {
		// globalTicksPerSecondCounter.start();
	}

	private StopWatch stopWatch = new StopWatch();
	private long ticks = 0;
	private long overallTicks = 0;
	private int elapsedSeconds = 0;

	@Override
	public void run() {
		while (!isInterrupted())
			try {
				stopWatch.reset();
				ticks = 0;
				stopWatch.start();
				Thread.sleep(1000);
				stopWatch.stop();
				overallTicks += ticks;
				elapsedSeconds++;
				System.out.println(ticks + " ticks/s (average: " + overallTicks / elapsedSeconds + " ticks/s)");
			}
			catch (InterruptedException e) {
				interrupt();
			}
	}

	public void tick() {
		ticks++;
	}

	public long getOverallTicks() {
		return overallTicks;
	}
}
