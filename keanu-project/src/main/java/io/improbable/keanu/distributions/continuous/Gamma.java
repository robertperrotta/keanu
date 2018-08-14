package io.improbable.keanu.distributions.continuous;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import static io.improbable.keanu.distributions.dual.Diffs.A;
import static io.improbable.keanu.distributions.dual.Diffs.K;
import static io.improbable.keanu.distributions.dual.Diffs.THETA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Gamma implements ContinuousDistribution {

    private static final double M_E = 0.577215664901532860606512090082;
    private final DoubleTensor location;
    private final DoubleTensor scale;
    private final DoubleTensor alpha;

    /**
     * <h3>Gamma Distribution</h3>
     *
     * @param location shifts the distribution
     * @param scale    stretches/shrinks the distribution, must be greater than 0
     * @param alpha    shape parameter (not to be confused with tensor shape)
     * @see "Computer Generation of Statistical Distributions
     * by Richard Saucier,
     * ARL-TR-2168 March 2000,
     * 5.1.11 page 23"
     */
    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale, DoubleTensor alpha) {
        return new Gamma(location, scale, alpha);
    }

    private Gamma(DoubleTensor location, DoubleTensor scale, DoubleTensor alpha) {
        this.location = location;
        this.scale = scale;
        this.alpha = alpha;
    }

    /**
     * @throws IllegalArgumentException if scale or alpha passed to {@link #withParameters(DoubleTensor location, DoubleTensor scale, DoubleTensor alpha)}
     *                                  is less than or equal to 0
     */
    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> locationWrapped = location.getFlattenedView();
        Tensor.FlattenedView<Double> scaleWrapped = scale.getFlattenedView();
        Tensor.FlattenedView<Double> alphaWrapped = alpha.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        double[] samples = new double[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(locationWrapped.getOrScalar(i), scaleWrapped.getOrScalar(i), alphaWrapped.getOrScalar(i), random);
        }

        return DoubleTensor.create(samples, shape);
    }

    private static double sample(double location, double scale, double alpha, KeanuRandom random) {
        if (scale <= 0. || alpha <= 0.) {
            throw new IllegalArgumentException("Invalid value for scale or alpha. Scale: " + scale + ". Alpha: " + alpha);
        }
        final double A = 1. / sqrt(2. * alpha - 1.);
        final double B = alpha - log(4.);
        final double Q = alpha + 1. / A;
        final double T = 4.5;
        final double D = 1. + log(T);
        final double C = 1. + alpha / M_E;

        if (alpha < 1.) {
            return sampleWhileKLessThanOne(C, alpha, location, scale, random);
        } else if (alpha == 1.0) {
            return exponentialSample(location, scale, random);
        } else {
            while (true) {
                double p1 = random.nextDouble();
                double p2 = random.nextDouble();
                double v = A * log(p1 / (1. - p1));
                double y = alpha * exp(v);
                double z = p1 * p1 * p2;
                double w = B + Q * v - y;
                if (w + D - T * z >= 0. || w >= log(z)) return location + scale * y;
            }
        }
    }

    private static double sampleWhileKLessThanOne(double c, double alpha, double location, double scale, KeanuRandom random) {
        while (true) {
            double p = c * random.nextDouble();
            if (p > 1.) {
                double y = -log((c - p) / alpha);
                if (random.nextDouble() <= pow(y, alpha - 1.)) return location + scale * y;
            } else {
                double y = pow(p, 1. / alpha);
                if (random.nextDouble() <= exp(-y)) return location + scale * y;
            }
        }
    }

    private static double exponentialSample(double location, double alpha, KeanuRandom random) {
        return location - alpha * Math.log(random.nextDouble());
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor aMinusXOverScale = location.minus(x).divInPlace(scale);
        final DoubleTensor alphaLnScale = alpha.times(scale.log());
        final DoubleTensor xMinusAPowAlphaMinus1 = x.minus(location).powInPlace(alpha.minus(1.));
        final DoubleTensor lnXMinusAToAlphaMinus1 = ((xMinusAPowAlphaMinus1).divInPlace(alpha.apply(org.apache.commons.math3.special.Gamma::gamma))).logInPlace();
        return aMinusXOverScale.minusInPlace(alphaLnScale).plusInPlace(lnXMinusAToAlphaMinus1);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor xMinusLocation = x.minus(location);
        final DoubleTensor locationMinusX = location.minus(x);
        final DoubleTensor alphaMinus1 = alpha.minus(1.);
        final DoubleTensor oneOverScale = scale.reciprocal();

        final DoubleTensor dLogPdlocation = alphaMinus1.div(locationMinusX).plusInPlace(oneOverScale);
        final DoubleTensor dLogPdscale = scale.times(alpha).plus(locationMinusX).divInPlace(scale.pow(2.)).unaryMinusInPlace();
        final DoubleTensor dLogPdalpha = xMinusLocation.logInPlace().minusInPlace(scale.log()).minusInPlace(alpha.apply(org.apache.commons.math3.special.Gamma::digamma));
        final DoubleTensor dLogPdx = alphaMinus1.div(xMinusLocation).minusInPlace(oneOverScale);

        return new Diffs()
        .put(A, dLogPdlocation)
        .put(THETA, dLogPdscale)
        .put(K, dLogPdalpha)
        .put(X, dLogPdx);
    }

}