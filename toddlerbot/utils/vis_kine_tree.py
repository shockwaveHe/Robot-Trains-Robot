import argparse
import textwrap
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import networkx as nx
import yaml
from networkx.drawing.nx_agraph import graphviz_layout


def add_nodes_and_edges(links, graph):
    """Add nodes and edges to the graph."""
    for link_name, link_info in links.items():
        graph.add_node(link_name, shape="box")
        for joint in link_info.get("joints", []):
            child_link = joint["child"]
            graph.add_node(
                child_link, shape="box"
            )  # Ensure child link is also added as a node
            graph.add_edge(link_name, child_link, label=joint["name"])


def parse_urdf_to_links_and_joints(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for joint in root.findall("joint"):
        parent_link = joint.find("parent").attrib["link"]
        child_link = joint.find("child").attrib["link"]
        joint_name = joint.attrib["name"]
        joint_type = joint.attrib["type"]  # Store the joint type

        if parent_link not in links:
            links[parent_link] = {"joints": []}
        # Include the joint type in the stored information
        links[parent_link]["joints"].append(
            {"name": joint_name, "child": child_link, "type": joint_type}
        )

        if child_link not in links:
            links[child_link] = {"joints": []}
    return links


def parse_yaml_to_links_and_joints(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("links", {})


def vis_kine_tree(
    links_and_joints,
    output_path=None,
    figsize=(10, 10),
    fixed_node_size=3000,
    font_size=6,
):
    G = nx.DiGraph()
    add_nodes_and_edges(links_and_joints, G)

    plt.figure(figsize=figsize)
    pos = graphviz_layout(G, prog="dot")

    nx.draw_networkx_nodes(
        G, pos, node_shape="o", node_size=fixed_node_size, node_color="skyblue"
    )
    wrapped_labels = {node: textwrap.fill(node, width=20) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=font_size)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=20, edge_color="gray")

    # Define colors for different joint types
    joint_colors = {
        "fixed": "blue",
        "revolute": "green",
        "prismatic": "red",
        # Add more joint types and colors if needed
    }

    # Create a legend for the joint types
    legend_labels = {
        jtype: plt.Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linestyle="None",
            markersize=10,
            label=jtype,
        )
        for jtype, color in joint_colors.items()
    }

    # Draw edge labels with colors based on joint type
    edge_labels = nx.get_edge_attributes(G, "label")
    for (start, end), label in edge_labels.items():
        joint_info = links_and_joints[start]["joints"]
        joint_type = next(
            (item for item in joint_info if item["child"] == end), {}
        ).get("type", "unknown")
        edge_color = joint_colors.get(
            joint_type, "black"
        )  # Default color if type is not in joint_colors
        x_start, y_start = pos[start]
        x_end, y_end = pos[end]
        label_x, label_y = (x_start + x_end) / 2, (y_start + y_end) / 2

        plt.text(
            label_x,
            label_y,
            label,
            color=edge_color,
            rotation="horizontal",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=font_size,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
        )

    plt.legend(handles=legend_labels.values(), loc="upper left")

    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a kinematics tree from a YAML or URDF file."
    )
    parser.add_argument("--path", type=str, help="Path to a YAML or URDF file.")
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Path to save the visualization as a PNG file.",
        default=None,
    )
    args = parser.parse_args()

    if args.path.endswith(".yaml") or args.path.endswith(".yml"):
        links_and_joints = parse_yaml_to_links_and_joints(args.path)
    elif args.path.endswith(".urdf"):
        links_and_joints = parse_urdf_to_links_and_joints(args.path)
    else:
        raise ValueError("Unsupported file format. Please provide a YAML or URDF file.")

    vis_kine_tree(links_and_joints, args.output_path)
