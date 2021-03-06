package io.improbable.keanu.vertices.dbl.probabilistic;

import static io.improbable.keanu.distributions.dual.Diffs.LAMBDA;
import static io.improbable.keanu.distributions.dual.Diffs.X;
import static io.improbable.keanu.tensor.TensorShape.shapeToDesiredRankByPrependingOnes;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar;

import java.util.Map;

import io.improbable.keanu.distributions.continuous.Exponential;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ExponentialVertex extends DoubleVertex implements ProbabilisticDouble {

    private final DoubleVertex lambda;

    /**
     * Lambda driving an arbitrarily shaped tensor of Exponential
     * <p>
     * pdf = lambda * exp(-lambda*x)
     * <p>
     * If all provided parameters are scalar then the proposed shape determines the shape
     *
     * @param tensorShape the desired shape of the vertex
     * @param lambda      the lambda of the Exponential with either be the same shape as specified for this
     *                    vertex or scalar.
     */
    public ExponentialVertex(int[] tensorShape, DoubleVertex lambda) {

        checkTensorsMatchNonScalarShapeOrAreScalar(tensorShape, lambda.getShape());

        this.lambda = lambda;
        setParents(lambda);
        setValue(DoubleTensor.placeHolder(tensorShape));
    }

    /**
     * One to one constructor for mapping some shape of lambda to matching shaped exponential.
     *
     * @param lambda the lambda of the Exponential with either the same shape as specified for this vertex or scalar
     */
    public ExponentialVertex(DoubleVertex lambda) {
        this(checkHasSingleNonScalarShapeOrAllScalar(lambda.getShape()), lambda);
    }

    public ExponentialVertex(double lambda) {
        this(new ConstantDoubleVertex(lambda));
    }

    @Override
    public double logProb(DoubleTensor value) {

        DoubleTensor lambdaValues = lambda.getValue();

        DoubleTensor logPdfs = Exponential.withParameters(lambdaValues).logProb(value);

        return logPdfs.sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(DoubleTensor value) {
        Diffs dlnP = Exponential.withParameters(lambda.getValue()).dLogProb(value);
        return convertDualNumbersToDiff(dlnP.get(LAMBDA).getValue(), dlnP.get(X).getValue());
    }

    private Map<Long, DoubleTensor> convertDualNumbersToDiff(DoubleTensor dLogPdlambda,
                                                             DoubleTensor dLogPdx) {

        PartialDerivatives dLogPdInputs = lambda.getDualNumber().getPartialDerivatives().multiplyBy(dLogPdlambda);

        if (!this.isObserved()) {
            dLogPdInputs.putWithRespectTo(getId(), dLogPdx.reshape(
                shapeToDesiredRankByPrependingOnes(dLogPdx.getShape(), dLogPdx.getRank() + getValue().getRank()))
            );
        }

        PartialDerivatives summed = dLogPdInputs.sum(true, TensorShape.dimensionRange(0, getShape().length));
        return summed.asMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return Exponential.withParameters(lambda.getValue()).sample(getShape(), random);
    }

}


