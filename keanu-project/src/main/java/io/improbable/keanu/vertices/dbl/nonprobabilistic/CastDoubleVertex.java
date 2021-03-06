package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class CastDoubleVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private final Vertex<? extends NumberTensor> inputVertex;

    public CastDoubleVertex(Vertex<? extends NumberTensor> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return inputVertex.sample(random).toDouble();
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().toDouble();
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        throw new UnsupportedOperationException("CastDoubleTensorVertex is non-differentiable");
    }
}
