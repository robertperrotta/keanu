package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.SimpleBooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Function;

import static java.util.Arrays.copyOf;

public class Nd4jIntegerTensor implements IntegerTensor {

    public static Nd4jIntegerTensor scalar(int scalarValue) {
        return new Nd4jIntegerTensor(Nd4j.scalar(scalarValue));
    }

    public static Nd4jIntegerTensor create(int[] values, int[] shape) {
        return new Nd4jIntegerTensor(values, shape);
    }

    public static Nd4jIntegerTensor create(double value, int[] shape) {
        return new Nd4jIntegerTensor(Nd4j.valueArrayOf(shape, value));
    }

    public static Nd4jIntegerTensor ones(int[] shape) {
        return new Nd4jIntegerTensor(Nd4j.ones(shape));
    }

    public static Nd4jIntegerTensor zeros(int[] shape) {
        return new Nd4jIntegerTensor(Nd4j.zeros(shape));
    }

    private INDArray tensor;
    private int[] shape;

    public Nd4jIntegerTensor(int[] data, int[] shape) {
        DataBuffer buffer = Nd4j.createBuffer(toFloat(data));
        this.tensor = Nd4j.create(buffer, shape);
        this.shape = shape;
    }

    public Nd4jIntegerTensor(int[] shape) {
        this.tensor = null;
        this.shape = shape;
    }

    public Nd4jIntegerTensor(INDArray tensor) {
        this.tensor = tensor;
        this.shape = tensor.shape();
    }

    private float[] toFloat(int[] data) {
        float[] floatData = new float[data.length];

        for (int i = 0; i < floatData.length; i++) {
            floatData[i] = data[i];
        }

        return floatData;
    }

    @Override
    public IntegerTensor minus(int value) {
        return duplicate().minusInPlace(value);
    }

    @Override
    public IntegerTensor plus(int value) {
        return duplicate().plusInPlace(value);
    }

    @Override
    public IntegerTensor times(int value) {
        return duplicate().timesInPlace(value);
    }

    @Override
    public IntegerTensor div(int value) {
        return duplicate().divInPlace(value);
    }

    @Override
    public IntegerTensor pow(IntegerTensor exponent) {
        return duplicate().powInPlace(exponent);
    }

    @Override
    public IntegerTensor pow(int exponent) {
        return duplicate().powInPlace(exponent);
    }

    @Override
    public IntegerTensor minus(IntegerTensor that) {
        return duplicate().minusInPlace(that);
    }

    @Override
    public IntegerTensor plus(IntegerTensor that) {
        return duplicate().plusInPlace(that);
    }

    @Override
    public IntegerTensor times(IntegerTensor that) {
        return duplicate().timesInPlace(that);
    }

    @Override
    public IntegerTensor div(IntegerTensor that) {
        return duplicate().divInPlace(that);
    }

    @Override
    public IntegerTensor unaryMinus() {
        return duplicate().unaryMinusInPlace();
    }

    @Override
    public IntegerTensor abs() {
        return duplicate().absInPlace();
    }

    @Override
    public IntegerTensor setWithMask(IntegerTensor mask, int value) {
        return duplicate().setWithMaskInPlace(mask, value);
    }

    @Override
    public IntegerTensor getGreaterThanMask(IntegerTensor greaterThanThis) {

        INDArray mask = tensor.dup();

        if (greaterThanThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldGreaterThan(mask,
                    Nd4j.valueArrayOf(mask.shape(), greaterThanThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray greaterThanThisArray = unsafeGetNd4J(greaterThanThis);
            Nd4j.getExecutioner().exec(
                new OldGreaterThan(mask, greaterThanThisArray, mask, mask.length())
            );
        }

        return new Nd4jIntegerTensor(mask);
    }

    @Override
    public IntegerTensor getGreaterThanOrEqualToMask(IntegerTensor greaterThanOrEqualToThis) {

        INDArray mask = tensor.dup();

        if (greaterThanOrEqualToThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldGreaterThanOrEqual(mask,
                    Nd4j.valueArrayOf(mask.shape(), greaterThanOrEqualToThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray greaterThanThisArray = unsafeGetNd4J(greaterThanOrEqualToThis);
            Nd4j.getExecutioner().exec(
                new OldGreaterThanOrEqual(mask, greaterThanThisArray, mask, mask.length())
            );
        }

        return new Nd4jIntegerTensor(mask);
    }

    @Override
    public IntegerTensor getLessThanMask(IntegerTensor lessThanThis) {

        INDArray mask = tensor.dup();

        if (lessThanThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldLessThan(mask,
                    Nd4j.valueArrayOf(mask.shape(), lessThanThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray lessThanThisArray = unsafeGetNd4J(lessThanThis);
            Nd4j.getExecutioner().exec(
                new OldLessThan(mask, lessThanThisArray, mask, mask.length())
            );
        }

        return new Nd4jIntegerTensor(mask);
    }

    @Override
    public IntegerTensor getLessThanOrEqualToMask(IntegerTensor lessThanOrEqualToThis) {

        INDArray mask = tensor.dup();

        if (lessThanOrEqualToThis.isScalar()) {
            Nd4j.getExecutioner().exec(
                new OldLessThanOrEqual(mask,
                    Nd4j.valueArrayOf(mask.shape(), lessThanOrEqualToThis.scalar()),
                    mask,
                    mask.length()
                )
            );
        } else {
            INDArray lessThanOrEqualToThisArray = unsafeGetNd4J(lessThanOrEqualToThis);
            Nd4j.getExecutioner().exec(
                new OldLessThanOrEqual(mask, lessThanOrEqualToThisArray, mask, mask.length())
            );
        }

        return new Nd4jIntegerTensor(mask);
    }

    @Override
    public IntegerTensor setWithMaskInPlace(IntegerTensor mask, int value) {

        INDArray maskDup = unsafeGetNd4J(mask).dup();

        if (value == 0.0) {
            INDArray swapOnesForZeros = Nd4j.ones(tensor.shape()).subi(maskDup);
            tensor.muli(swapOnesForZeros);
        } else {
            Nd4j.getExecutioner().exec(
                new CompareAndSet(maskDup, value, Conditions.equals(1.0))
            );

            Nd4j.getExecutioner().exec(
                new CompareAndSet(tensor, maskDup, Conditions.notEquals(0.0))
            );
        }

        return this;
    }

    @Override
    public IntegerTensor apply(Function<Integer, Integer> function) {
        return duplicate().applyInPlace(function);
    }

    @Override
    public IntegerTensor minusInPlace(int value) {
        tensor.subi(value);
        return this;
    }

    @Override
    public IntegerTensor plusInPlace(int value) {
        tensor.addi(value);
        return this;
    }

    @Override
    public IntegerTensor timesInPlace(int value) {
        tensor.muli(value);
        return this;
    }

    @Override
    public IntegerTensor divInPlace(int value) {
        Transforms.floor(tensor.divi(value), false);
        return this;
    }

    @Override
    public IntegerTensor powInPlace(IntegerTensor exponent) {
        if (exponent.isScalar()) {
            Transforms.pow(tensor, exponent.scalar(), false);
        } else {
            INDArray exponentArray = unsafeGetNd4J(exponent);
            Transforms.pow(tensor, exponentArray, false);
        }
        return this;
    }

    @Override
    public IntegerTensor powInPlace(int exponent) {
        Transforms.pow(tensor, exponent, false);
        return this;
    }

    @Override
    public IntegerTensor minusInPlace(IntegerTensor that) {

        if (that.isScalar()) {
            minusInPlace(that.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(that);
            tensor.subi(indArray);
        }

        return this;
    }

    @Override
    public IntegerTensor plusInPlace(IntegerTensor that) {

        if (that.isScalar()) {
            plusInPlace(that.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(that);
            tensor.addi(indArray);
        }

        return this;
    }

    @Override
    public IntegerTensor timesInPlace(IntegerTensor that) {

        if (that.isScalar()) {
            timesInPlace(that.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(that);
            tensor.muli(indArray);
        }

        return this;
    }

    @Override
    public IntegerTensor divInPlace(IntegerTensor that) {

        if (that.isScalar()) {
            divInPlace(that.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(that);
            Transforms.floor(tensor.divi(indArray), false);
        }

        return this;
    }

    @Override
    public IntegerTensor unaryMinusInPlace() {
        tensor.negi();
        return this;
    }

    @Override
    public IntegerTensor absInPlace() {
        Transforms.abs(tensor, false);
        return this;
    }

    @Override
    public IntegerTensor applyInPlace(Function<Integer, Integer> function) {
        DataBuffer data = tensor.data();
        for (int i = 0; i < data.length(); i++) {
            data.put(i, function.apply(data.getInt(i)));
        }
        return this;
    }

    @Override
    public BooleanTensor lessThan(int value) {
        return fromMask(tensor.lt(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor lessThanOrEqual(int value) {
        return fromMask(tensor.lte(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor lessThan(IntegerTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lt(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.lt(indArray);
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor lessThanOrEqual(IntegerTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lte(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.dup();
            Nd4j.getExecutioner().exec(new OldLessThanOrEqual(mask, indArray, mask, getLength()));
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThan(int value) {
        return fromMask(tensor.gt(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(int value) {
        return fromMask(tensor.gte(value), copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThan(IntegerTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gt(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.gt(indArray);
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(IntegerTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gt(value.scalar());
        } else {
            INDArray indArray = unsafeGetNd4J(value);
            mask = tensor.dup();
            Nd4j.getExecutioner().exec(new OldGreaterThanOrEqual(mask, indArray, mask, getLength()));
        }

        return fromMask(mask, copyOf(shape, shape.length));
    }

    @Override
    public Integer sum() {
        return tensor.sumNumber().intValue();
    }

    @Override
    public DoubleTensor toDouble() {
        return new Nd4jDoubleTensor(tensor.dup());
    }

    @Override
    public IntegerTensor toInteger() {
        return this;
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public long getLength() {
        return tensor.lengthLong();
    }

    @Override
    public boolean isShapePlaceholder() {
        return tensor == null;
    }

    @Override
    public Integer getValue(int... index) {
        return tensor.getInt(index);
    }

    @Override
    public void setValue(Integer value, int... index) {
        tensor.putScalar(index, value);
    }

    @Override
    public Integer scalar() {
        return tensor.getInt(0);
    }

    @Override
    public IntegerTensor duplicate() {
        return new Nd4jIntegerTensor(tensor.dup());
    }

    @Override
    public FlattenedView<Integer> getFlattenedView() {
        return new Nd4jIntegerFlattenedView(tensor);
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor<Integer> that) {

        if (that instanceof Nd4jIntegerTensor) {
            INDArray eq = tensor.eq(unsafeGetNd4J((Nd4jIntegerTensor) that));
            return fromMask(eq, shape);
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    private INDArray unsafeGetNd4J(IntegerTensor that) {
        return ((Nd4jIntegerTensor) that).tensor;
    }

    private BooleanTensor fromMask(INDArray mask, int[] shape) {
        DataBuffer data = mask.data();
        boolean[] boolsFromMask = new boolean[mask.length()];

        for (int i = 0; i < boolsFromMask.length; i++) {
            boolsFromMask[i] = data.getInt(i) != 0;
        }
        return new SimpleBooleanTensor(boolsFromMask, shape);
    }

    private static class Nd4jIntegerFlattenedView implements FlattenedView<Integer> {

        INDArray tensor;

        public Nd4jIntegerFlattenedView(INDArray tensor) {
            this.tensor = tensor;
        }

        @Override
        public long size() {
            return tensor.data().length();
        }

        @Override
        public Integer get(long index) {
            return tensor.data().getInt(index);
        }

        @Override
        public Integer getOrScalar(long index) {
            if (tensor.isScalar()) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, Integer value) {
            tensor.data().put(index, value);
        }

        @Override
        public double[] asDoubleArray() {
            return tensor.data().asDouble();
        }

        @Override
        public int[] asIntegerArray() {
            return tensor.data().asInt();
        }

        @Override
        public Integer[] asObjectArray() {
            return ArrayUtils.toObject(asIntegerArray());
        }
    }

}
