package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class GenericPluckVertex<T> extends UnaryOpVertex<Tensor<T>,Tensor<T>> {

    private final Vertex<? extends Tensor<T>> inputVertex;
    private final int[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex
     * @param index       the index of extraction
     */
    public GenericPluckVertex(Vertex<Tensor<T>> inputVertex, int... index) {
        super(inputVertex);
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
        this.inputVertex = inputVertex;
        this.index = index;
        setParents(inputVertex);
        setValue(Tensor.placeHolder(Tensor.SCALAR_SHAPE));
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    protected Tensor<T> op(Tensor<T> input) {
        return Tensor.scalar(input.getValue(index));
    }

}
