package com.example.baseball

import io.improbable.keanu.algorithms.mcmc.Hamiltonian
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings
import io.improbable.keanu.algorithms.mcmc.NUTS
import io.improbable.keanu.network.BayesianNetwork
import io.improbable.keanu.tensor.dbl.DoubleTensor
import io.improbable.keanu.tensor.intgr.IntegerTensor
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.KeanuRandom
import io.improbable.keanu.vertices.dbl.probabilistic.BetaVertex
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
import io.improbable.keanu.vertices.intgr.IntegerVertex
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex
import krangl.*
import java.io.File

fun main(args: Array<String>) {

    val data = DataFrame.readTSV(File("keanu-examples/baseball/src/main/resources/efron-morris-75-data.tsv"))

    val hits = data["Hits"].asInts().filterNotNull().toIntArray()
    val atBats = data["At-Bats"].asInts().filterNotNull().toIntArray()

    val phi = UniformVertex(0.0, 1.0)

    // We don't yet have a ParetoVertex but the following is equivalent.
    val kBase = ExponentialVertex(1.0)
    val kappa = kBase.exp().times(1.5)

    val thetas = BetaVertex(intArrayOf(1, data.nrow), kappa.times(phi), kappa.times(phi.unaryMinus().plus(1.0)))
    val ys = MyBinomialVertex(thetas, ConstantIntegerVertex(atBats))

    ys.observe(hits)

    val bayesNet = BayesianNetwork(ys.connectedGraph)

    val shuffle = { listOf<DoubleVertex>(phi, kBase, thetas).forEach { it.setAndCascade(it.sample()) } }

    shuffle()
    val mhSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
        bayesNet, bayesNet.latentVertices, 10_000).drop(2_000).downSample(4)

    val mhMeanPhi = mhSamples.get(phi).asList().map { it.scalar() } .mean()
    val mhMeanKappa = mhSamples.get(kBase).asList().map { it.exp().times(1.5).scalar() } .mean()
    println("MH mean of phi: $mhMeanPhi, Mean of kappa: $mhMeanKappa")

    shuffle()
    val hmcSamples = Hamiltonian.getPosteriorSamples(bayesNet, bayesNet.latentVertices,
        1_000, 10, 0.1).drop(200)

    val hmcMeanPhi = hmcSamples.get(phi).asList().map { it.scalar() } .mean()
    val hmcMeanKappa = mhSamples.get(kBase).asList().map { it.exp().times(1.5).scalar() } .mean()
    println("HMC mean of phi: $hmcMeanPhi, Mean of kappa: $hmcMeanKappa")

    shuffle()
    val nutsSamples = try {
        NUTS.getPosteriorSamples(bayesNet, bayesNet.latentVertices, 1_000,
            10, 0.99, KeanuRandom.getDefaultRandom())
            .drop(200)
    } catch(e: StackOverflowError) {
        println("${phi.value.scalar()}, ${kappa.value.scalar()}")
        throw e
    }

    val nutsMeanPhi = nutsSamples.get(phi).asList().map { it.scalar() } .mean()
    val nutsMeanKappa = nutsSamples.get(kBase).asList().map { it.exp().times(1.5).scalar() } .mean()
    println("NUTS mean of phi: $nutsMeanPhi, Mean of kappa: $nutsMeanKappa")

}

// Extension of BinomialVertex that calculates dLogProb with respect to p assuming n and k are fixed.
// (i.e. constant n and observed k)
class MyBinomialVertex(private val p: DoubleVertex, private val n: IntegerVertex): BinomialVertex(p, n) {

    override fun dLogProb(value: IntegerTensor): MutableMap<Long, DoubleTensor> {
        if (!isObserved || n !is ConstantIntegerVertex) {
            throw IllegalStateException("Can only calculate dLogProb if n is constant and k is observed!")
        }
        
        val pVal = p.value
        val nVal = n.value.toDouble()
        val kVal = value.toDouble()

        val out = mutableMapOf<Long, DoubleTensor>()

        // k / p - (n - k) / (1 - p) = k / p + (n - k) / (p - 1)
        out[p.id] = kVal.div(pVal).plusInPlace(nVal.minus(kVal).divInPlace(pVal.minus(1.0)))

        return out
    }

}
