# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
# 2025: Modified for MadNLP.jl and UnoSolver.jl by Alexis Montoison
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# The QP block used to be implemented here; it now lives in MathOptInterface,
# together with the evaluator layers that compose it with the
# vector-nonlinear-oracle constraints and the nonlinear model.
const QPBlockData = MOI.Nonlinear.QPBlockData
