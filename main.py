# Procedures Imports
import math
from matplotlib import pyplot
import numpy
import algos.utils
import algos.numerical.diff_integration
import algos.search
import algos.search.stochastic.random
from algos.graph import UndirectedGraph, DirectedGraph, WeightedGraph, WeightedEdge

import scipy
# Structures Imports
from algos.array import Array
from algos.graph import Graph

import platform

print("########## START MAIN CLIENT ########## \n")

print("########## SYSTEM INFORMATION ##########")
print("########## ARCHITECTURE:", platform.processor())
print("########## OPERATING SYSTEM:", platform.system())
print("######################################## \n")

###########################################################################
# TODO: ORGANIZE UTILS AND  BASIC STRUCTURES
# VARIABLES AND STRUCTURES FOR TESTING ALGOS
# array = Array(3)
# array.set_element(0, 'first')
# array.render()
#
# g = Graph(5)
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 3)
# g.add_edge(3, 4)
# g.render()
#
# # TODO: CHECK THIS SEARCH RESULT AND PATH
# dfs = algos.search.DepthFirstSearch(g, 4)
# is_path = dfs.has_path_to(g, 3)
# path = dfs.path_to(g, 4)
# print(path.elements)
# print(is_path)
#
###########################################################################


###########################################################################
# SEARCH ALGOS
# Linear search
# l = algos.search.linear(cl, 666, 0, len(cl) - 1)
# ls = algos.search.linear_sentinel(cl, 666, 0, len(cl) - 1)
# lo = algos.search.linear_ordered(cl, 666)
#
# # Binary search
# b = algos.search.binary(cl, 666, 0, len(cl) - 1)
# bb = algos.search.binary_ordered(cl, 666, 0, len(cl))
# bl = algos.search.binary_left(cl, 666, 0, len(cl) - 1)
# br = algos.search.binary_right(cl, 666, 0, len(cl) - 1)
# r = algos.search.binary_recursive(cl, 666, 0, len(cl) - 1)
#
# print("########## SEARCH INDEX RESULT:", l, ls, lo, b, bb, bl, br, r)
# print(cl[18])

# Random Search
# problem_size = 2
# search_space = [-5, 5]
# max_iter = 100
#
# best = algos.search.stochastic.random.random_search(search_space, max_iter)
# print('Done. Best Solution: c = ', best['cost'], 'v = ', best['vector'])

g = UndirectedGraph()
g.add_edge(4, 2)
g.add_edge(2, 1)
g.add_edge(3, 2)
g.add_edge(1, 5)

print('Vertices:', g.vertices())
print('Edges:', g.edges())

we = WeightedEdge(1, 2, 1.3)
wg = WeightedGraph(5)

wg.add_edge(we)
###########################################################################


###########################################################################
# NUMERIC ALGOS
# # Original Function
# f = lambda x: (x ** 3) + (4 * (x ** 2)) - 10
# # Algebraic Equivalent function to f
# g = lambda x: x - (((x ** 3) + (4 * (x ** 2)) - 10) / (3 * (x ** 2) + (8 * x)))

# Simpson rule
# 3.c EXERCITE
# f_I = lambda x: 2 / ((x ** 2) + 4)
# n_I = 6
# a_I = 0
# b_I = 2

# 3.b
# f_I = lambda x: (x ** 3) * (math.exp(x))
# n_I = 4
# a_I = -2
# b_I = 2

# Romberg
# 1.a
f_I = lambda x: math.sin(x)
n_I = 6
a_I = 0
b_I = math.pi

# List of polynomials coefficients
# p = (-10, 0, 4, 1)
# # function polynomial
# px = lambda x: (x ** 4) - (3 * (x ** 3)) + (x ** 2) + x + 1
#
# # Data for interpolation
# x_nums = [1.0, 1.3, 1.6, 1.9, 2.2]
# f_values = [
#     0.7651977,
#     0.6200860,
#     0.4554022,
#     0.2818186,
#     0.1103623,
# ]
#
# h_nums = [1.3, 1.6, 1.9]
# h_values = [
#     0.6200860,
#     0.4554022,
#     0.2818186
# ]
#
# h_df = [
#     -0.5220232,
#     -0.5698959,
#     -0.5811571
# ]
# nat_nums = [0, 1, 2, 3]
# nat_values = [1, math.e, math.e ** 2, math.e ** 3]
#
# clap_nums = [0, 1, 2, 3]
# clap_values = [1, math.e, math.e ** 2, math.e ** 3]
#
# # Bezier Curve 3.a
# point_0 = (1, 1)
# point_1 = (6, 2)
# left_plus_points = (1.5, 1.25)
# right_minus_points = (7, 3)

# 3.d
# p_n = [
#     (0, 0),
#     (2, 1),
#     (4, 0),
#     (6, -1),
# ]
# g_0 = (0.5, 0.5)
# g_n = (6.5, -0.25)
# g_l_points = [
#     (3, 1),
#     (5, 1),
# ]
# g_r_points = [
#     (3, 1),
#     (3, -1),
# ]

# # Bisection Method
# f_bisection = algos.numerical.bisection(f, 1, 2, 0.005, 10)
#
# # Fixed Point Method
# g_fixed_point = algos.numerical.fixed_point_iteration(g, 1.5, 0.0001, 20)
#
# # Newton Method
# f_newton = algos.numerical.newton(f, 1.5, 0.005, 10)
#
# # Secant Method
# f_secant = algos.numerical.secant(f, 1, 2, 0.005, 10)
#
# # False Method
# f_false_position = algos.numerical.false_position(f, 1, 2, 0.005, 10)
#
# # Steffensen Method
# g_steffensen = algos.numerical.steffensen(g, 1.5, 0.0001, 20)
#
# Horner Method
# p_horner = algos.numerical.horner(len(p) - 1, p, 2)
#
# # Muller Method
# px_muller = algos.numerical.muller(px, 0.5, -0.5, 0, 0.00001, 9)
#
# # Neville Method
# p_neville = algos.numerical.neville(f_values, x_nums, 1.5)
#
# # Newton's Divided Difference Method
# p_newton_divided_diff = algos.numerical.newton_divided_difference(f_values, x_nums)
#
# # Hermite Interpolation
# h_hemite = algos.numerical.hermite_interpolation(h_values, h_df, h_nums)

# Natural Cubic Spline Interpolation
# nat_cubic = algos.numerical.natural_cubic_spline(nat_values, nat_nums)
#
# # Clamped Cubic Spline Interpolation
# clamped_cubic = algos.numerical.clamped_cubic_spline(clap_values, clap_nums, 1, math.e ** 3)

# # Cubic Bezier Curve
# bezier_c = algos.numerical.bezier(point_0, point_1, left_plus_points, right_minus_points)
# bezier_n = algos.numerical.bezier_curve(p_n, g_r_points, g_l_points, g_0, g_n)
#
# # Casteljau Method
# x_domain = numpy.arange(0, 1, 0.01)
# y_range = algos.numerical.casteljau(x_domain, bezier_c)
#
# pyplot.plot(x_domain, y_range)
# pyplot.show()

# composite simpson rule
# rule = algos.numerical.composite_simpsons_rule(f_I, a_I, b_I, n_I)

# Romberg Method
# romberg = algos.numerical.diff_integration.romberg(f_I, a_I, b_I, n_I)

# Adaptive Quadrature
f = lambda x: (100 / numpy.power(x, 2)) * numpy.sin(10.0 / x)
a = 1.0
b = 3.0
tol = numpy.power(10.0, -4)
n = 23

# adapt = algos.numerical.diff_integration.quad_asr(f, a, b, tol)
# print(adapt)
# adapt = algos.numerical.diff_integration.adaptive_quadrature(f, a, b, n, tol)
# print(adapt)
# adapt = scipy.integrate.quadrature(f, a, b)

# print(adapt)

###########################################################################

print("########## END MAIN CLIENT ##########")
