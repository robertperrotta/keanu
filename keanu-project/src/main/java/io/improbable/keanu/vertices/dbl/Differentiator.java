package io.improbable.keanu.vertices.dbl;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class Differentiator {
    private Differentiator() {}

    public static <V extends Vertex & Differentiable> DualNumber calculateDual(V vertex) {
        Map<Vertex, DualNumber> dualNumbers = new HashMap<>();
        Deque<V> stack = new ArrayDeque<>();
        stack.push(vertex);

        while (!stack.isEmpty()) {

            V head = stack.peek();
            Set<Vertex> parentsThatDualNumberIsNotCalculated = parentsThatDualNumberIsNotCalculated(dualNumbers, head.getParents());

            if (parentsThatDualNumberIsNotCalculated.isEmpty()) {
                V top = stack.pop();
                DualNumber dual = top.calculateDualNumber(dualNumbers);
                dualNumbers.put(top, dual);
            } else {
                for (Vertex parent : parentsThatDualNumberIsNotCalculated) {
                    if (parent instanceof Differentiable) {
                        stack.push((V) parent);
                    }
                }
            }
        }

        return dualNumbers.get(vertex);
    }

    private static Set<Vertex> parentsThatDualNumberIsNotCalculated(Map<Vertex, DualNumber> dualNumbers, Collection<? extends Vertex> parents) {
        Set<Vertex> notCalculatedParents = new HashSet<>();
        for (Vertex next : parents) {
            if (!dualNumbers.containsKey(next) && next instanceof Differentiable) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }
}
