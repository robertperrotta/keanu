package com.example.baseball

import io.improbable.keanu.algorithms.mcmc.MetropolisHastings
import io.improbable.keanu.algorithms.mcmc.NUTS
import io.improbable.keanu.network.BayesianNetwork
import io.improbable.keanu.vertices.dbl.KeanuRandom
import io.improbable.keanu.vertices.dbl.probabilistic.BetaVertex
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
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
    val kappa = ExponentialVertex(1.0).exp().times(1.5)

    val thetas = BetaVertex(intArrayOf(1, data.nrow), kappa.times(phi), kappa.times(phi.unaryMinus().plus(1.0)))
    val ys = BinomialVertex(thetas, ConstantIntegerVertex(atBats))

    ys.observe(hits)

    val bayesNet = BayesianNetwork(ys.connectedGraph)

    val samples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
        bayesNet, bayesNet.latentVertices, 2000)

    // NOT YET SUPPORTED! Requires differentiation over p on BinomialVertex (an extension of SOL-1975)
    // val samples = NUTS.getPosteriorSamples(bayesNet, bayesNet.latentVertices,
    //         2000, 10, 0.99, KeanuRandom.getDefaultRandom())

}
