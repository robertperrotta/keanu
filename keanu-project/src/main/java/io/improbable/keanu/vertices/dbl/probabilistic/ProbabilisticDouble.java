package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;

public interface ProbabilisticDouble extends Probabilistic<DoubleTensor> {
    default double logPdf(double value) {
        return logPdf(DoubleTensor.scalar(value));
    }

    default double logPdf(double[] values) {
        return logPdf(DoubleTensor.create(values));
    }

    default double logPdf(DoubleTensor value) {
        return logProb(value);
    }

    default Map<Long, DoubleTensor> dLogPdf(double value) {
        return dLogPdf(DoubleTensor.scalar(value));
    }

    default Map<Long, DoubleTensor> dLogPdf(double[] values) {
        return dLogPdf(DoubleTensor.create(values));
    }

    default Map<Long,DoubleTensor> dLogPdf(DoubleTensor value) {
        return dLogProb(value);
    }
}
