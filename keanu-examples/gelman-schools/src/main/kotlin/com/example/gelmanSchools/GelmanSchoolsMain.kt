package com.example.gelmanSchools

import io.improbable.keanu.algorithms.mcmc.NUTS
import io.improbable.keanu.network.BayesianNetwork
import io.improbable.keanu.vertices.dbl.KeanuRandom
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex
import io.improbable.keanu.vertices.dbl.probabilistic.HalfCauchyVertex

fun main(args: Array<String>) {

    val y = doubleArrayOf(28.0,  8.0, -3.0,  7.0, -1.0,  1.0, 18.0, 12.0)
    val sigma = doubleArrayOf(15.0, 10.0, 16.0, 11.0,  9.0, 11.0, 10.0, 18.0)

    val eta = GaussianVertex(intArrayOf(1, y.size), 0.0, 1.0)
    val mu = GaussianVertex(0.0, 1e6)
    val tau = HalfCauchyVertex(25.0)

    val theta = mu + tau * eta

    val obs = GaussianVertex(theta, ConstantDoubleVertex(sigma))
    obs.observe(y)

    val bayesNet = BayesianNetwork(obs.connectedGraph)

    val samples = NUTS.getPosteriorSamples(bayesNet, bayesNet.latentVertices,
        1000, 10, 0.90, KeanuRandom.getDefaultRandom())

}
