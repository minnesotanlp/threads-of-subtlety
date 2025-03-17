import os

import networkx as nx

from tos.tos_utils import load_graph_motifs, save_graph_motifs

motifs_dir = "data/motifs"
single_motifs_path = os.path.join(motifs_dir, "hc3-mage_M3_motifs.json")
double_motifs_path = os.path.join(motifs_dir, "hc3-mage_M6_motifs.json")
single_motifs = load_graph_motifs(single_motifs_path)
double_motifs = load_graph_motifs(double_motifs_path)
single_motifs = {
    h: nx.convert_node_labels_to_integers(m) for h, m in single_motifs.items()
}
double_motifs = {
    h: nx.convert_node_labels_to_integers(m) for h, m in double_motifs.items()
}
print(f"no. of single motifs: {len(single_motifs)}")
print(f"no. of double motifs: {len(double_motifs)}")

# leave only triangular motifs
single_triangular_motifs = {
    k: v for k, v in single_motifs.items() if v.number_of_edges() == 3
}
double_triangular_motifs = {
    k: v for k, v in double_motifs.items() if v.number_of_edges() == 6
}

# identify node to attach single motif, relabel it as 'docking_point'
filter_double_motif_hashes = set()
for double_m_h, double_m_g in double_triangular_motifs.items():
    docking_point_found = False
    for double_m_node in double_m_g.nodes():
        if (
            double_m_g.in_degree(double_m_node) == 2
            and double_m_g.out_degree(double_m_node) == 0
        ):
            nx.relabel_nodes(double_m_g, {double_m_node: "docking_point"}, copy=False)
            docking_point_found = True
            break
    if not docking_point_found:
        filter_double_motif_hashes.add(double_m_h)
for filter_hash in filter_double_motif_hashes:
    del double_triangular_motifs[filter_hash]

# label each one of two non-root nodes as 'docking_point'
single_motifs_prepared = {}
for single_m_h, single_m_g in single_triangular_motifs.items():
    nx.relabel_nodes(single_m_g, {0: "a", 1: "b", 2: "c"}, copy=False)
    candidates = []
    for single_m_node in single_m_g.nodes():
        if (
            single_m_g.in_degree(single_m_node) == 1
            and single_m_g.out_degree(single_m_node) == 1
        ):
            candidates.append(
                nx.relabel_nodes(
                    single_m_g, {single_m_node: "docking_point"}, copy=True
                )
            )
        elif (
            single_m_g.in_degree(single_m_node) == 0
            and single_m_g.out_degree(single_m_node) == 2
        ):
            candidates.append(
                nx.relabel_nodes(
                    single_m_g, {single_m_node: "docking_point"}, copy=True
                )
            )
    if len(candidates) > 0:
        single_motifs_prepared[single_m_h] = candidates

triple_motifs_candidates = {}
for double_m_h, double_m_g in double_triangular_motifs.items():
    for single_m_h, single_m_gs in single_motifs_prepared.items():
        for single_m_g in single_m_gs:
            triple_motif = nx.compose(double_m_g, single_m_g)
            triple_hash = nx.weisfeiler_lehman_graph_hash(
                triple_motif, edge_attr="label_0"
            )
            triple_motifs_candidates[triple_hash] = triple_motif
print(f"len of triple triangular motifs candidates: {len(triple_motifs_candidates)}")
triple_motifs_candidates_path = "hc3-mage_M9_motifs.json"
save_graph_motifs(9, triple_motifs_candidates.values(), "hc3-mage", show_tracking=True)
