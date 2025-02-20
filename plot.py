import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def plot_top10_gene_sensitivity(save,occlu_deta_score, xlabel, ylabel, filename, linewidth=1.5, figsize=(16, 10),
                                    color='silver'):
    gene_list = sorted(occlu_deta_score.items(), key=lambda item: item[1], reverse=True)

    sorted_names = [i[0] for i in gene_list[:10]]
    sorted_scores = [i[1] for i in gene_list[:10]]
    data = {"sorted_names": sorted_names, "sorted_scores": sorted_scores}
    data = pd.DataFrame(data)

    figure, ax = plt.subplots(figsize=figsize, dpi=100)
    sns.barplot(data=data, x='sorted_names', y='sorted_scores', facecolor=color, linewidth=linewidth)

    plt.tick_params(labelsize=35)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    plt.xticks(rotation=45)

    font1 = { 'weight': 'normal', 'size': 35, }
    ax.set_xlabel(xlabel, font1)
    ax.set_ylabel(ylabel, font1)

    figure.subplots_adjust(right=0.9)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.31)
    plt.savefig(save)
    plt.show()
        
def plot_all_gene_sensitivity(occlu_deta_score,save):
    gene_list = sorted(occlu_deta_score.items(), key=lambda item: item[1], reverse=True)

    sorted_names = [i[0] for i in gene_list]
    sorted_scores = [i[1] for i in gene_list]
    plt.bar(sorted_names, sorted_scores)
    plt.xticks([])
    plt.yticks(fontsize=40)
    plt.xlabel("all genes", fontsize = 40)
    plt.ylabel("sensitivity", fontsize = 40)
    plt.tight_layout()
    plt.savefig(save)
    plt.show()

