package io.improbable.keanu.research.vertices;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

public class DoubleBinaryOpLambda<A, B> extends NonProbabilisticDouble {

    protected final Vertex<A> a;
    protected final Vertex<B> b;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<Vertex, DualNumber>, DualNumber> dualNumberSupplier;

    public DoubleBinaryOpLambda(int[] shape,
                                Vertex<A> a,
                                Vertex<B> b,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation) {
        this.a = a;
        this.b = b;
        this.op = op;
        this.dualNumberSupplier = dualNumberCalculation;
        setParents(a, b);
    }

    public DoubleBinaryOpLambda(int[] shape, Vertex<A> a, Vertex<B> b, BiFunction<A, B, DoubleTensor> op) {
        this(shape, a, b, op, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(a.sample(random), b.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op.apply(a.getValue(), b.getValue());
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}