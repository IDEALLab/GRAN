import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re

type_to_label = {'R': 0, 'L': 1, 'C': 2, 'D': 3, 'V': 4, 'S': 5, 'joint': 6, 'VP': 7, 'nonjoint': 8}
type_to_color = {
    'R': 'b',
    'L': 'g',
    'C': 'r',
    'D': 'yellow',
    'V': 'orange',
    'S': 'purple',
    'joint': 'gray',
    'VP': 'orange',
    'nonjoint': 'gray'
}


def draw_graph_list(G_list,
                    row,
                    col,
                    fname='exp/gen_graph.png',
                    layout='spring',
                    is_single=False,
                    k=1,
                    node_size=55,
                    alpha=1,
                    width=1.3):
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.axis("off")

        # turn off axis label
        plt.xticks([])
        plt.yticks([])

        if layout == 'spring':
            pos = nx.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes(
                G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0, font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699', alpha=1, linewidths=0.2, font_size=1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def draw_graph_list_separate(G_list,
                             fname='exp/gen_graph',
                             layout='spring',
                             is_single=False,
                             k=1,
                             node_size=55,
                             alpha=1,
                             width=1.3):

    for i, G in enumerate(G_list):
        print('i =', i)
        plt.switch_backend('agg')

        plt.axis("off")

        # turn off axis label
        # plt.xticks([])
        # plt.yticks([])

        if layout == 'spring':
            pos = nx.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)

        if is_single:
            # # node_size default 60, edge_width default 1.5
            # nx.draw_networkx_nodes(
            #     G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0, font_size=0)
            # nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
            '========XD========'
            print('G.nodes:', G.nodes())
            node_colors = [type_to_color[rm_idx(n)] for n in G.nodes()]
            print('node_colors:', node_colors)
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=200, font_size=10)
            '========XD========'
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699', alpha=1, linewidths=0.2, font_size=1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

        plt.draw()
        plt.tight_layout()
        plt.savefig(fname + '_{:03d}.png'.format(i), dpi=300)
        plt.close()


def rm_idx(s):
    # turns 'VS0' into 'VS'
    # todo: how to split 'non-joint0'? that '-'
    # print(s, [i for i in re.split(r'([A-Za-z]+)', s) if i])
    e_type, idx = [i for i in re.split(r'([A-Za-z]+)', s) if i]
    return e_type