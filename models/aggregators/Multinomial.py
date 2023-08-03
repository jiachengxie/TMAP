import random

def multinomial_sample(num_samples, probabilities):
    """
    从Multinomial分布中采样。

    :param num_samples: 采样的总次数（实验次数）
    :param probabilities: 每个结果的概率列表，概率之和应为1
    :return: 采样结果，表示每个结果出现的次数的列表
    """
    num_categories = len(probabilities)
    samples = [0] * num_categories

    remaining_samples = num_samples
    for i in range(num_categories - 1):
        # 使用二项分布采样每个结果的次数
        count = random.randint(0, remaining_samples)
        samples[i] = count
        remaining_samples -= count

    # 最后一个结果的次数等于剩余次数
    samples[num_categories - 1] = remaining_samples

    return samples


# # 定义每个结果的概率
# probs = [0.3, 0.5, 0.2]
#
# # 定义实验次数
# num_samples = 10
#
# # 从Multinomial分布中采样
# samples = multinomial_sample(num_samples, probs)
#
# print(samples)