import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


class KruskalWallis:
    def __init__(self, data: dict):
        """
        Initializes the Scott-Knott class.

        :param {dict} data - A dictionary whose keys are the names of
        different methods to compare, and the values are results. The
        values must be list-like objects.
        """
        self.data = data

    def pprint(self):
        # Perform Kruskal-Wallis test
        _, p_value = stats.kruskal(*self.data.values())
        group_medians = {key: np.median(val) for key, val in self.data.items()}
        print(f"Group medians: {group_medians}")
        print(f"Kruskal-Wallis test p-value: {p_value}")

        if p_value < 0.05:
            # Find the group with the largest median
            max_group = max(group_medians, key=group_medians.get)
            print(f"Group with the largest median: {max_group}")

            # Perform pairwise Mann-Whitney U tests
            groups = list(self.data.keys())
            num_groups = len(groups)
            p_values = np.zeros((num_groups, num_groups))

            for i in range(num_groups):
                for j in range(i + 1, num_groups):
                    _, p = stats.mannwhitneyu(self.data[groups[i]], self.data[groups[j]], alternative='less')
                    p_values[i, j] = p
                    p_values[j, i] = p

            print('Pairwise Mann-Whitney U tests')
            print(pd.DataFrame(p_values, index=groups, columns=groups))
            print()

            # Apply Bonferroni correction for multiple comparisons
            adjusted_p_values = multipletests(p_values.ravel(), method='fdr_tsbh')[1].reshape(p_values.shape)
            post_hoc = pd.DataFrame(adjusted_p_values, index=groups, columns=groups)

            print("Pairwise Mann-Whitney U tests with Benjamini/Hochberg correction:")
            print(post_hoc)
            print()

            # Check if the group with the largest median is significantly better than the others
            significantly_better = True
            for key in self.data.keys():
                if key != max_group and post_hoc.loc[max_group, key] >= 0.05:
                    significantly_better = False
                    break

            if significantly_better:
                print(f"The group '{max_group}' with the largest median IS significantly better than the others, ",
                      end='')
            else:
                print("The group with the largest median IS NOT significantly better than all the others, ", end='')

            print(f'and the highest p-value is {round(max(post_hoc[max_group]), 2)}')

            return post_hoc, max_group, significantly_better
        else:
            print("There is no significant difference among the groups.")
            return None, None, None
