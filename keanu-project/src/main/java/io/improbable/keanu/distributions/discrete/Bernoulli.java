package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Bernoulli implements Distribution<BooleanTensor> {

    private final DoubleTensor probOfEvent;

    /**
     * <h3>Bernoulli Distribution</h3>
     *
     * @param probOfEvent probability of an event
     * @see "Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.2.1 page 42"
     */
    public static Distribution<BooleanTensor> withParameters(DoubleTensor probOfEvent) {
        return new Bernoulli(probOfEvent);
    }

    private Bernoulli(DoubleTensor probOfEvent) {
        this.probOfEvent = probOfEvent;
    }

    @Override
    public BooleanTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor uniforms = random.nextDouble(shape);
        return uniforms.lessThan(probOfEvent);
    }

    @Override
    public DoubleTensor logProb(BooleanTensor x) {
        DoubleTensor probOfEventClamped = probOfEvent.clamp(DoubleTensor.ZERO_SCALAR, DoubleTensor.ONE_SCALAR);

        DoubleTensor probability = x.setDoubleIf(
            probOfEventClamped,
            probOfEventClamped.unaryMinus().plusInPlace(1.0)
        );

        return probability.logInPlace();
    }

}