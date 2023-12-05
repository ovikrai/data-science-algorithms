import numpy as np
import math
import random


def upper_confidence_bound(data: np.ndarray,
                           n=10000,
                           d=10):
    items_selected = []
    numbers_of_selections = [0] * d
    sums_of_rewards = [0] * d
    total_reward = 0

    for j in range(0, n):
        item = 0
        max_upper_bound = 0
        for i in range(0, d):
            if numbers_of_selections[i] > 0:
                average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                delta_i = math.sqrt(3 / 2 * math.log(j + 1) / numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                item = i
        items_selected.append(item)
        numbers_of_selections[item] = numbers_of_selections[item] + 1
        reward = data.values[j, item]
        sums_of_rewards[item] = sums_of_rewards[item] + reward
        total_reward = total_reward + reward

    return items_selected


def thompson_sampling(data: np.ndarray,
                      n=10000,
                      d=10):
    items_selected = []
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d
    total_reward = 0
    for j in range(0, n):
        item = 0
        max_random = 0
        for i in range(0, d):
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                             numbers_of_rewards_0[i] + 1)
            if random_beta > max_random:
                max_random = random_beta
                item = i
        items_selected.append(item)
        reward = data.values[j, item]
        if reward == 1:
            numbers_of_rewards_1[item] = numbers_of_rewards_1[item] + 1
        else:
            numbers_of_rewards_0[item] = numbers_of_rewards_0[item] + 1
        total_reward = total_reward + reward

    return items_selected
