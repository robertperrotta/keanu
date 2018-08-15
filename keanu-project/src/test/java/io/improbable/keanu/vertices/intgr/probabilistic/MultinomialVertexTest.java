package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.both;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import static io.improbable.keanu.tensor.TensorMatchers.allCloseTo;
import static io.improbable.keanu.tensor.TensorMatchers.allValues;
import static io.improbable.keanu.tensor.TensorMatchers.hasShape;
import static io.improbable.keanu.tensor.TensorMatchers.hasValue;

import java.util.Map;

import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.distributions.discrete.Binomial;
import io.improbable.keanu.distributions.discrete.Categorical;
import io.improbable.keanu.distributions.discrete.Multinomial;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ReshapeVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class MultinomialVertexTest {

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheProbabilitiesDontSumToOne() {
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0., 0., 0.99, 0.).transpose();
        Multinomial.withParameters(n, p);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheParametersAreDifferentShapes() {
        IntegerTensor n = IntegerTensor.create(100, 200);
        DoubleTensor p = DoubleTensor.create(0., 0., 1., 0.).transpose();
        Multinomial.withParameters(n, p);
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheSampleShapeDoesntMatchTheShapeOfN() {
        IntegerTensor n = IntegerTensor.create(100, 200);
        DoubleTensor p = DoubleTensor.create(new double[]{
            0.1, 0.25,
            0.2, 0.25,
            0.3, 0.25,
            0.4, 0.25
        },
            4, 2);
        Multinomial multinomial = Multinomial.withParameters(n, p);
        multinomial.sample(new int[]{2, 2}, KeanuRandom.getDefaultRandom());
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbShapeDoesntMatchTheNumberOfCategories() {
        IntegerTensor n = IntegerTensor.create(100);
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).transpose();
        Multinomial multinomial = Multinomial.withParameters(n, p);
        multinomial.logProb(IntegerTensor.scalar(1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbStateDoesntSumToN() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        multinomial.logProb(IntegerTensor.create(5, 6).transpose());
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbStateContainsNegativeNumbers() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        multinomial.logProb(IntegerTensor.create(-1, 11).transpose());
    }

    @Test(expected = IllegalArgumentException.class)
    public void itThrowsIfTheLogProbStateContainsNumbersGreaterThanN() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.3, 0.5).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        int[] state = new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE, 12};
        assertThat(state[0] + state[1] + state[2], equalTo(10));
        multinomial.logProb(IntegerTensor.create(state).transpose());
    }

    @Test
    public void itWorksWithScalars() {
        int n = 100;
        DoubleTensor p = DoubleTensor.create(0.01, 0.09, 0.9).transpose();
        MultinomialVertex multinomial = new MultinomialVertex(
            ConstantVertex.of(n), ConstantVertex.of(p));
        IntegerTensor samples = multinomial.sample(KeanuRandom.getDefaultRandom());
        assertThat(samples, hasShape(3, 1));
        assertThat(samples, allValues(both(greaterThan(-1)).and(lessThan(n))));
    }

    @Test
    public void itWorksWithTensors() {
        IntegerVertex n = ConstantVertex.of(IntegerTensor.create(new int[]{
            1, 10,
            100, 1000},
            2, 2));

        DoubleVertex p = ConstantVertex.of(DoubleTensor.create(new double[]{
                .1, .8,
                .25, .2,

                .1, .1,
                .50, .3,

                .8, .1,
                .25, .5
            },
            3, 2, 2));
        //
        MultinomialVertex multinomial = new MultinomialVertex(n, p);
        IntegerTensor sample = multinomial.sample(KeanuRandom.getDefaultRandom());
        assertThat(sample, hasShape(3, 2, 2));
        double logProb = multinomial.logProb(IntegerTensor.create(new int[]{
                0, 10,
                25, 200,

                0, 0,
                50, 300,

                1, 0,
                25, 500,
            },
            3, 2, 2));
        assertThat(logProb, closeTo(-14.165389164658901, 1e-8));
    }

    @Test
    public void youCanUseAConcatAndReshapeVertexToPipeInTheProbabilities() {
        IntegerVertex n = ConstantVertex.of(IntegerTensor.create(new int[]{
                1, 10,
                100, 1000},
            2, 2));

        DoubleVertex p1 = ConstantVertex.of(DoubleTensor.create(new double[]{
                .1, .8,
                .25, .2,
            },
            2, 2));

        DoubleVertex p2 = ConstantVertex.of(DoubleTensor.create(new double[]{
                .1, .1,
                .50, .3,
            },
            2, 2));

        DoubleVertex p3 = ConstantVertex.of(DoubleTensor.create(new double[]{

                .8, .1,
                .25, .5
            },
            2, 2));

        ConcatenationVertex pConcatenated = new ConcatenationVertex(0, p1, p2, p3);
        ReshapeVertex pReshaped = new ReshapeVertex(pConcatenated, 3, 2, 2);
        MultinomialVertex multinomial = new MultinomialVertex(n, pReshaped);
        IntegerTensor sample = multinomial.sample(KeanuRandom.getDefaultRandom());
        assertThat(sample, hasShape(3, 2, 2));
        double logProb = multinomial.logProb(IntegerTensor.create(new int[]{
                0, 10,
                25, 200,

                0, 0,
                50, 300,

                1, 0,
                25, 500,
            },
            3, 2, 2));

        assertThat(logProb, equalTo(-14.165389164658901));
    }


    @Test
    public void youCanSampleWithATensorIfNIsScalarAndPIsAColumnVector() {
        int n = 100;
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, .3, 0.4).transpose();
        Multinomial multinomial = Multinomial.withParameters(IntegerTensor.scalar(n), p);
        IntegerTensor samples = multinomial.sample(new int[]{2, 2}, KeanuRandom.getDefaultRandom());
        assertThat(samples, hasShape(4, 2, 2));
        assertThat(samples, allValues(both(greaterThan(-1)).and(lessThan(n))));
    }
    
    @Test
    public void ifTheresOnlyOneValidChoiceItAlwaysReturnsIt() {
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0., 0., 1., 0.).transpose();
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new int[]{1, 1}, KeanuRandom.getDefaultRandom());
        assertThat(samples, hasValue(0, 0, 100, 0));
    }

    @Test
    public void ifYourRandomReturnsZeroItSamplesFromTheFirstNonZeroCategory() {
        KeanuRandom mockRandomAlwaysZero = mock(KeanuRandom.class);
        when(mockRandomAlwaysZero.nextDouble()).thenReturn(0.);
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0., 0.5, .5, 0.).transpose();
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new int[]{1, 1}, mockRandomAlwaysZero);
        assertThat(samples, hasValue(0, 100, 0, 0));
    }

    @Test
    public void ifYourRandomReturnsOneItSamplesFromTheLastNonZeroCategory() {
        KeanuRandom mockRandomAlwaysZero = mock(KeanuRandom.class);
        when(mockRandomAlwaysZero.nextDouble()).thenReturn(1.);
        IntegerTensor n = IntegerTensor.scalar(100);
        DoubleTensor p = DoubleTensor.create(0., 0.5, .5, 0.).transpose();
        Multinomial multinomial = Multinomial.withParameters(n, p);
        IntegerTensor samples = multinomial.sample(new int[]{1, 1}, mockRandomAlwaysZero);
        assertThat(samples, hasValue(0, 0, 100, 0));
    }

    @Test
    public void whenKEqualsTwoItsBinomial() {
        IntegerTensor n = IntegerTensor.scalar(10);
        DoubleTensor p = DoubleTensor.create(0.2, 0.8).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);
        DiscreteDistribution binomial = Binomial.withParameters(DoubleTensor.scalar(0.2), n);
        for (int value : ImmutableList.of(1, 2, 9, 10)) {
            DoubleTensor binomialLogProbs = binomial.logProb(IntegerTensor.scalar(value));
            DoubleTensor multinomialLogProbs = multinomial.logProb(IntegerTensor.create(value, 10 - value).transpose()).transpose();
            assertThat(multinomialLogProbs, allCloseTo(new Double(1e-6), binomialLogProbs));
        }
    }

    enum Colours {
        RED, GREEN, BLUE
    }

    @Test
    public void whenKNEqualsOneItsCategorical() {
        IntegerTensor n = IntegerTensor.scalar(1);
        DoubleTensor p = DoubleTensor.create(0.2, .3, 0.5).transpose();
        DiscreteDistribution multinomial = Multinomial.withParameters(n, p);

        Map<Colours, DoubleVertex> selectableValues = ImmutableMap.of(
            Colours.RED, ConstantVertex.of(p.getValue(0)),
            Colours.GREEN, ConstantVertex.of(p.getValue(1)),
            Colours.BLUE, ConstantVertex.of(p.getValue(2)));
        Categorical categorical = Categorical.withParameters(selectableValues);

        double pRed = categorical.logProb(Colours.RED);
        assertThat(multinomial.logProb(IntegerTensor.create(1, 0, 0).transpose()).scalar(), closeTo(pRed, 1e-7));
        double pGreen = categorical.logProb(Colours.GREEN);
        assertThat(multinomial.logProb(IntegerTensor.create(0, 1, 0).transpose()).scalar(), closeTo(pGreen, 1e-7));
        double pBlue = categorical.logProb(Colours.BLUE);
        assertThat(multinomial.logProb(IntegerTensor.create(0, 0, 1).transpose()).scalar(), closeTo(pBlue, 1e-7));
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 10000;
        DoubleTensor p = DoubleTensor.create(0.1, 0.2, 0.3, 0.4).transpose();
        IntegerTensor n = IntegerTensor.scalar(500);

        MultinomialVertex vertex = new MultinomialVertex(
            new int[]{1, N},
            ConstantVertex.of(n),
            ConstantVertex.of(p)
        );

        IntegerTensor samples = vertex.sample();
        assertThat(samples, hasShape(4, N));

        for (int i = 0; i < samples.getShape()[0]; i++) {
            System.out.println(i);
            IntegerTensor sample = samples.slice(0, i);
            Double probability = p.slice(0, i).scalar();
            double mean = sample.toDouble().average();
            double std = sample.toDouble().standardDeviation();

            double epsilonForMean = 0.5;
            double epsilonForVariance = 5.;
            assertEquals(n.scalar() * probability, mean, epsilonForMean);
            assertEquals(n.scalar() * probability * (1 - probability), std * std, epsilonForVariance);
        }
    }
}