## Setup
import pandas as pd
import numpy as np

import re
import csv
import itertools
import json
from collections import defaultdict

import scipy
import scipy.stats
import statsmodels
import statsmodels.stats.multitest

import seaborn as sns
import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib_venn as mpvenn
import matplotlib.ticker as ticker
import matplotlib as mpl
import scikitplot as skplt

from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Fix random seed for reproducability
SEED = 1
np.random.seed(SEED)

# Plot defaults
sns.set()
params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (15, 10),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "figure.titlesize": "xx-large",
    "font.family": "Helvetica",
}
plt.rcParams.update(params)


## Data Import & Pre-processing


def load_raw_data():

    # Load CSV files
    kegg_b = (
        pd.read_csv("./data/metagenomics/kegg_combined_baseline_stripped.T.csv")
        .set_index("Index")
        .T
    )

    kegg_e = (
        pd.read_csv("./data/metagenomics/kegg_combined_end_stripped.T.csv")
        .set_index("Index")
        .T
    )

    metadata = pd.read_csv("./data/metagenomics/metadata.csv").set_index("Key")

    # Filter to only keep samples with metagenomic data
    metadata = metadata[metadata.HasMetagenomics]

    # Fix column/index names
    kegg_b.columns.name = kegg_e.columns.name = ""
    kegg_b.index.name = kegg_e.index.name = "Key"

    # Remove period qualifiers from metadata indices and kegg indices
    def remove_period_qualifier(i):
        if ".P" in i:
            i = i.split(".")
            return ".".join([i[0]] + i[2:])
        return i

    metadata.index = metadata.index.map(remove_period_qualifier)
    kegg_b.index = kegg_b.index.map(remove_period_qualifier)
    kegg_e.index = kegg_e.index.map(remove_period_qualifier)

    # Create corresponding metadata dataframe indices from KEGG dataframes
    kegg_b_index = kegg_b.index.map(
        lambda i: f"{i.split('.')[0]}.Baseline.{i.split('.')[-1]}"
    )
    kegg_e_index = kegg_e.index.map(
        lambda i: f"{i.split('.')[0]}.End.{i.split('.')[-1]}"
    )

    assert all(kegg_b_index.isin(metadata.index))
    assert all(kegg_e_index.isin(metadata.index))

    # Subset metadata + remove "Baseline" and "End"
    metadata_b = metadata.loc[kegg_b_index, :]
    metadata_e = metadata.loc[kegg_e_index, :]
    metadata_b.index = metadata_b.index.map(lambda i: i.replace(".Baseline", ""))
    metadata_e.index = metadata_e.index.map(lambda i: i.replace(".End", ""))

    # Return baseline/end kegg and metadata dataframes
    return kegg_b, kegg_e, metadata_b, metadata_e


def get_ko_name(koid):
    if not hasattr(get_ko_name, "mapping"):
        get_ko_name.mapping = json.load(open("./data/other/kegg_ko_name_mapping.json"))
    if koid in get_ko_name.mapping:
        return get_ko_name.mapping[koid]
    else:
        return "N/A"


def get_gene_pathways(koid):
    if not hasattr(get_gene_pathways, "gene_kegg_links"):
        gene_kegg_links = pd.read_csv("./data/other/gene_kegg_links.csv").set_index(
            "GeneID"
        )
        gene_kegg_links["PathwayID"] = gene_kegg_links["PathwayID"].map(
            lambda i: i[5:].upper().replace("KO", "K")
        )

        get_gene_pathways.gene_kegg_links = gene_kegg_links

    if koid not in get_gene_pathways.gene_kegg_links.index:
        return "N/A"

    pathway_ids = get_gene_pathways.gene_kegg_links.loc[koid, :].values.flatten()

    return list(pathway_ids)


def get_gene_brite(koid, levels=["H3"]):
    if not hasattr(get_gene_brite, "gene_brite_map"):
        gene_brite_map = pd.read_csv("./data/other/kegg_brite_mapping.csv").set_index(
            "KO"
        )

        get_gene_brite.gene_brite_map = gene_brite_map

    if koid not in get_gene_brite.gene_brite_map.index:
        return ["Unclassified"]

    brite = get_gene_brite.gene_brite_map.loc[[koid], levels]
    merged = list(brite[levels].apply(lambda x: ";; ".join(x), axis=1).values)
    return merged


def get_pathway_names(koids):
    if not hasattr(get_pathway_names, "pathway_names"):
        pathway_names = pd.read_csv("./data/other/pathway_names.csv").set_index(
            "PathwayID"
        )
        pathway_names.index = pathway_names.index.map(
            lambda i: i[5:].upper().replace("KO", "K")
        )
        pathway_names.loc["N/A"] = ["N/A"]

        get_pathway_names.pathway_names = pathway_names

    names = get_pathway_names.pathway_names.loc[koids, :].values.flatten()

    return list(names)


def log_normalize(df):
    # Normalize data using log transform
    return np.log2(1 + df)


def relative_abundance(df):
    return df.div(df.sum(axis=1), axis=0)


def calculate_difference(kegg_b, kegg_e):
    return kegg_e - kegg_b


## Differential Gene Expression

# Calculating DGE
def calc_dge(q=0.2):

    kegg_b, kegg_e, metadata_b, metadata_e = load_raw_data()
    kegg_b, kegg_e = log_normalize(kegg_b), log_normalize(kegg_e)
    kegg_d = calculate_difference(kegg_b, kegg_e)

    dge = {}

    for study in sorted(set(metadata_e.Study)):

        # Subset to control and treatment differences
        kegg_dfc = kegg_d.loc[
            (metadata_e.Study == study) & (metadata_e.Treatment == "Control")
        ]
        kegg_dft = kegg_d.loc[
            (metadata_e.Study == study) & (metadata_e.Treatment == study)
        ]

        # Calculate fold change values for each of the genes
        kegg_dft_mean = kegg_dft.mean(axis=0)
        kegg_dfc_mean = kegg_dfc.mean(axis=0)
        kegg_diff_fc = kegg_dft_mean - kegg_dfc_mean

        # Compute p-values using t-test column-wise
        _, ps = scipy.stats.ttest_ind(kegg_dfc, kegg_dft, axis=0)

        # Extract feature names
        cols = kegg_d.columns

        # Subset both ps and features to where p is not nan
        cols = cols[pd.notna(ps)]
        ps = ps[pd.notna(ps)]

        # FDR correct
        _, psc, _, _ = statsmodels.stats.multitest.multipletests(
            ps, alpha=q, method="fdr_bh"
        )

        # Find columns where psc < alpha
        idx = np.argwhere(psc < q).flatten()

        psc = psc[idx]
        genes = cols[idx]
        kegg_diff_fc = kegg_diff_fc[genes]
        kegg_dft_mean = kegg_dft_mean[genes]
        kegg_dfc_mean = kegg_dfc_mean[genes]

        dge[study] = {
            "p": psc,
            "genes": genes,
            "log2fc": kegg_diff_fc,
            "treatment-mean-diff": kegg_dft_mean,
            "control-mean-diff": kegg_dfc_mean,
        }

    return dge


dge = calc_dge()


# Exporting DGE
def export_dge_for_kegga_and_pathview():
    kegg_b, kegg_e, metadata_b, metadata_e = load_raw_data()

    dge = calc_dge()

    # Write universe to file
    universe = list(kegg_e.columns)
    universe = [[x] for x in universe]
    with open("./kegga/universe.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(["KO"])
        write.writerows(universe)

    # Write DGEs to CSVs
    for food in dge.keys():
        df = pd.DataFrame(dge[food]).set_index("genes")
        df = df.sort_values("p")

        if len(df) == 0:
            continue

        df.to_csv(f"./kegga/dge_{food}.csv")

        df["name"] = df.index.map(get_ko_name)
        df["pathways"] = df.index.map(
            lambda ko: ";; ".join(get_pathway_names(get_gene_pathways(ko)))
        )
        df["brite"] = df.index.map(
            lambda ko: "; ".join(get_gene_brite(ko, levels=["H2", "H3"]))
        )

        df.to_csv(f"./kegga/dge_{food}_with_names.csv")


export_dge_for_kegga_and_pathview()


# Filtering DGE
def filter_dge_by_lfc(dge, min_lfc):
    dge = dge.copy()

    for food in dge.keys():

        log2fc = dge[food]["log2fc"]
        genes = dge[food]["genes"]

        idx = np.argwhere(np.abs(log2fc.values) >= min_lfc).flatten()

        dge[food]["p"] = dge[food]["p"][idx]
        dge[food]["genes"] = dge[food]["genes"][idx]
        dge[food]["log2fc"] = dge[food]["log2fc"][idx]

    return dge


# DGE Overlap
def dge_overlap():
    # Extract genes
    dge = calc_dge()
    genes_per_food = {
        food: set(dge[food]["genes"])
        for food in list(filter(lambda x: len(dge[x]["genes"]) != 0, dge.keys()))
    }

    # Plot
    foods = list(genes_per_food.keys())
    genes = list(genes_per_food.values())

    res = mpvenn.venn3_unweighted(genes, set_labels=foods)

    for text in res.set_labels:
        text.set_fontsize(18)
    for text in res.subset_labels:
        text.set_fontsize(16)

    plt.title("Overlap of Differentially Expressed Genes")
    plt.savefig("./figures/dge_overlap.png", bbox_inches="tight")
    plt.close()

    # Build dictionary of overlaps for manual inspection

    Almond = genes_per_food["Almond"]
    Broccoli = genes_per_food["Broccoli"]
    Walnut = genes_per_food["Walnut"]
    overlap = {
        "Almond": Almond - (Broccoli | Walnut),
        "Broccoli": Broccoli - (Almond | Walnut),
        "Almond-Broccoli": (Almond & Broccoli) - Walnut,
        "Walnut": Walnut - (Almond | Broccoli),
        "Almond-Walnut": (Almond & Walnut) - Broccoli,
        "Broccoli-Walnut": (Broccoli & Walnut) - Almond,
        "Almond-Broccoli-Walnut": Almond & Broccoli & Walnut,
    }

    for key in overlap:
        df = pd.DataFrame(overlap[key], columns=["KO"])
        df = df.set_index("KO")

        df["name"] = df.index.map(get_ko_name)
        df["pathways"] = df.index.map(
            lambda ko: ";; ".join(get_pathway_names(get_gene_pathways(ko)))
        )
        df["brite"] = df.index.map(
            lambda ko: "; ".join(get_gene_brite(ko, levels=["H2", "H3"]))
        )

        df.to_csv(f"./results/dge_overlap_{key}.csv")

    return overlap


overlap = dge_overlap()


## High Variance Plots
def high_variance_plots():

    kegg_b, kegg_e, metadata_b, metadata_e = load_raw_data()

    # Keep track of all features
    all_features = set()

    S = 20  # number of features to export
    n_features = 20  # number of features to plot
    assert S >= n_features

    for idx, study in enumerate(sorted(list(set(metadata_e.Study)))):

        kegg_b, kegg_e, metadata_b, metadata_e = load_raw_data()
        kegg_bn, kegg_en = log_normalize(kegg_b), log_normalize(kegg_e)

        # Find features with high variance across the food (using log transform data)
        kegg_all = pd.concat(
            [kegg_bn.filter(like=study, axis=0), kegg_en.filter(like=study, axis=0)]
        )
        kegg_top = kegg_all.var().sort_values(ascending=False)[:S]

        # Add features list to all features set
        all_features.update(list(kegg_top.index))

        # Subset to however many features we want
        kegg_top = kegg_top[:n_features]

        features_list = list(kegg_top.index)

        # Subset to control/treatment, and top columns
        kegg_bc = kegg_b.loc[
            (metadata_b.Study == study) & (metadata_b.Treatment == "Control"),
            features_list,
        ]
        kegg_bt = kegg_b.loc[
            (metadata_b.Study == study) & (metadata_b.Treatment == study), features_list
        ]
        kegg_ec = kegg_e.loc[
            (metadata_e.Study == study) & (metadata_e.Treatment == "Control"),
            features_list,
        ]
        kegg_et = kegg_e.loc[
            (metadata_e.Study == study) & (metadata_e.Treatment == study), features_list
        ]

        # Compute relative abundances
        kegg_bc, kegg_bt, kegg_ec, kegg_et = (
            relative_abundance(kegg_bc),
            relative_abundance(kegg_bt),
            relative_abundance(kegg_ec),
            relative_abundance(kegg_et),
        )

        kegg_bca = pd.DataFrame(kegg_bc.mean(axis=0)).T
        kegg_bta = pd.DataFrame(kegg_bt.mean(axis=0)).T
        kegg_eca = pd.DataFrame(kegg_ec.mean(axis=0)).T
        kegg_eta = pd.DataFrame(kegg_et.mean(axis=0)).T

        # kegg_bca.index = ["Pre-intervention"]  # ["Before Control"]
        # kegg_eca.index = ["Post-intervention"]  # ["After Control"]
        # kegg_bta.index = ["Pre-intervention"]  # ["Before Treatment"]
        # kegg_eta.index = ["Post-intervention"]  # ["After Treatment"]
        kegg_bca.index = [f"{study} Control"]
        kegg_bta.index = [f"{study} Treatment"]
        kegg_eca.index = [f"{study} Control"]
        kegg_eta.index = [f"{study} Treatment"]

        kegg_a = pd.concat([kegg_bca, kegg_bta, kegg_eca, kegg_eta])

        diff = kegg_eta.iloc[0] - kegg_bta.iloc[0]
        keys = list(diff.sort_values().index)

        kegg_a = kegg_a.loc[:, keys]

        # map KOs to names
        kegg_a.columns = kegg_a.columns.map(lambda i: f"{get_gene_brite(i)[0]} ({i})")

        # plot

        palette = sns.color_palette(cc.glasbey_category10, n_colors=n_features)

        fig, ax = plt.subplots(figsize=(14, 10))
        kegg_a.plot.bar(
            stacked=True,
            legend=True,
            ax=ax,
            color=palette,
            edgecolor="none",
            width=0.45,
        )
        ax.legend(
            ncol=4,
            bbox_to_anchor=(0, -0.72, 1, 1),
            loc="center",
            borderaxespad=0,
            prop={"size": 11},
        )
        fig.suptitle("")

        # plot connecting lines
        for j in range(len(features_list)):
            segments = []
            for i in range(3):

                # skip second --> third bar
                if i == 1:
                    continue

                x1 = ax.containers[j][i].get_x() + ax.containers[j][i].get_width()
                y1 = ax.containers[j][i].get_y() + ax.containers[j][i].get_height()

                x2 = ax.containers[j][
                    i + 1
                ].get_x()  # + ax.containers[0][i].get_width()
                y2 = (
                    ax.containers[j][i + 1].get_y()
                    + ax.containers[j][i + 1].get_height()
                )
                segments.append([(x1, y1), (x2, y2)])

                collection = mc.LineCollection(
                    segments, linewidths=1, colors=palette[j]
                )
                ax.add_collection(collection)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)
        ax.xaxis.grid(False)
        ax.set_ylim([0, 1.075])

        # second axis labels
        ax2 = ax.twiny()
        ax2.xaxis.grid(False)

        ax2.spines["bottom"].set_position(("axes", -0.075))
        ax2.tick_params("both", length=0, width=0, which="minor", pad=8, labelsize=20)
        ax2.tick_params("both", direction="in", which="major", labelsize=18)
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")

        ax2.set_xticks([0.0, 0.5, 1.0])
        ax2.xaxis.set_major_formatter(ticker.NullFormatter())
        ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.25, 0.75]))
        ax2.xaxis.set_minor_formatter(
            ticker.FixedFormatter([f"Pre-intervention", f"Post-intervention"])
            # ticker.FixedFormatter([f"{study} Control", f"{study} Treatment"])
        )

        # other stuff
        letter = ["A", "B", "C", "D", "E"][idx]

        ax.set_ylabel("Relative KO abundance", size=20)

        ax.text(
            0.02,
            0.95,
            letter,
            transform=ax.transAxes,
            fontsize=24,
            fontweight=1000,
        )

        plt.savefig(f"./figures/high_variance_barplot_{study}.svg", bbox_inches="tight")
        plt.close()

    # Save high-variance features to file
    pd.DataFrame(index=list(all_features)).to_csv(
        "./figures/high_variance_features.csv"
    )

    # create additional plot of highest-variable features prior to intervention for all foods
    kegg_b, kegg_e, metadata_b, metadata_e = load_raw_data()
    kegg_bn = log_normalize(kegg_b)
    high_var_bi = list(kegg_bn.var().sort_values(ascending=False)[:S].index)
    high_var_bi = kegg_b[high_var_bi]
    high_var_bi = relative_abundance(high_var_bi)
    # map KOs to names
    high_var_bi.columns = high_var_bi.columns.map(
        lambda i: f"{get_gene_brite(i)[0]} ({i})"
    )

    high_var_bi = (
        metadata_b[["Study"]]
        .merge(high_var_bi, left_index=True, right_index=True)
        .groupby("Study")
        .mean()
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    palette = sns.color_palette(cc.glasbey_category10, n_colors=n_features)
    high_var_bi.plot.bar(
        stacked=True,
        legend=True,
        ax=ax,
        color=palette,
        edgecolor="none",
        width=0.45,
    )
    ax.legend(
        ncol=4,
        bbox_to_anchor=(0, -0.65, 1, 1),
        loc="center",
        borderaxespad=0,
        prop={"size": 11},
    )
    fig.suptitle("")
    ax.set_ylabel("Relative KO abundance", size=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)
    ax.set_ylim([0, 1.075])
    ax.xaxis.grid(False)
    ax.set_xlabel("")
    ax.text(
        0.02,
        0.95,
        "F",
        transform=ax.transAxes,
        fontsize=24,
        fontweight=1000,
    )
    plt.savefig(
        f"./figures/high_variance_barplot_before_interevention.svg", bbox_inches="tight"
    )
    plt.close()


high_variance_plots()


## Clustermap
def clusterMap(dataFrame, cellSizePixelsW=60, cellSizePixelsH=40):
    figHeight = 800
    cellSizePixelsH = figHeight // len(dataFrame)

    cellSizePixelsW = np.max([cellSizePixelsH, 60])

    # Calulate the figure size, this gets us close, but not quite to the right place
    dpi = matplotlib.rcParams["figure.dpi"]
    marginWidth = (
        matplotlib.rcParams["figure.subplot.right"]
        - matplotlib.rcParams["figure.subplot.left"]
    )
    marginHeight = (
        matplotlib.rcParams["figure.subplot.top"]
        - matplotlib.rcParams["figure.subplot.bottom"]
    )
    Ny, Nx = dataFrame.shape
    figWidth = (Nx * cellSizePixelsW / dpi) / 0.8 / marginWidth
    figHeigh = (Ny * cellSizePixelsH / dpi) / 0.8 / marginHeight

    # do the actual plot
    cmap = sns.diverging_palette(10, 150, l=35, as_cmap=True)

    grid = sns.clustermap(
        dataFrame,
        figsize=(figWidth, figHeigh),
        cmap=cmap,
        vmax=5,
        vmin=-5,
        center=0,
        linewidths=0.1,
        fmt=".1f",
        cbar_pos=None,
        yticklabels=True,
    )

    fig = plt.gcf()
    width_px, height_px = fig.get_size_inches() * fig.dpi

    # calculate the size of the heatmap axes
    axWidth = (Nx * cellSizePixelsW) / (figWidth * dpi)
    axHeight = (Ny * cellSizePixelsH) / (figHeigh * dpi)

    # resize heatmap
    ax_heatmap_orig_pos = grid.ax_heatmap.get_position()
    grid.ax_heatmap.set_position(
        [ax_heatmap_orig_pos.x0, ax_heatmap_orig_pos.y0, axWidth, axHeight]
    )

    # resize dendrograms to match
    ax_row_orig_pos = grid.ax_row_dendrogram.get_position()
    # grid.ax_row_dendrogram.set_position(
    #    [ax_row_orig_pos.x0, ax_row_orig_pos.y0, ax_row_orig_pos.width, axHeight]
    # )
    grid.ax_row_dendrogram.set_position(
        [
            ax_row_orig_pos.x0 - ax_row_orig_pos.width / 2,
            ax_row_orig_pos.y0,
            ax_row_orig_pos.width * 2,
            axHeight * 1,
        ]
    )

    ax_col_orig_pos = grid.ax_col_dendrogram.get_position()
    grid.ax_col_dendrogram.set_position(
        [
            ax_col_orig_pos.x0,
            ax_heatmap_orig_pos.y0 + axHeight,
            axWidth,
            50 / height_px,
        ]
    )

    grid.ax_heatmap.set_xticklabels(
        grid.ax_heatmap.get_xmajorticklabels(), fontsize=22, rotation=60
    )
    grid.ax_heatmap.set_yticklabels(
        grid.ax_heatmap.get_ymajorticklabels(), fontsize=20, rotation=0
    )

    return grid  # return ClusterGrid object


## Single-food classification
def single_food_classification():

    PLOT_FEATURE_IMPORTANCE = True
    LFC_FILTER = 2

    kegg_b, kegg_e, metadata_b, metadata_e = load_raw_data()
    kegg_b, kegg_e = log_normalize(kegg_b), log_normalize(kegg_e)
    kegg_d = calculate_difference(kegg_b, kegg_e)

    dge = calc_dge()
    if LFC_FILTER != 0:
        dge = filter_dge_by_lfc(dge, LFC_FILTER)

    best_features_all = {}

    # list of (food, [feature importances to plot]) so we can plot it on the same graph
    raw_imps = []

    for food in dge.keys():

        if len(dge[food]["genes"]) == 0:
            continue

        # Extract control + treatment set for this food
        kegg_food = kegg_d.filter(like=food, axis=0)
        food_control = kegg_food.filter(like=f"No{food}", axis=0)
        food_treatment = kegg_food.loc[~kegg_food.index.isin(food_control.index), :]

        # Subset to DGE
        food_control = food_control[dge[food]["genes"]]
        food_treatment = food_treatment[dge[food]["genes"]]

        # Perform LOO classification

        X = pd.concat([food_control, food_treatment])
        y_true = ([f"No{food}"] * len(food_control)) + ([food] * len(food_treatment))

        clf = RandomForestClassifier(
            n_estimators=2000,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=SEED,
            verbose=0,
            class_weight="balanced",
        )

        clf.fit(X, y_true)  # fit so we get class assignments

        y_proba = cross_val_predict(
            clf, X, y_true, cv=LeaveOneOut(), n_jobs=-1, method="predict_proba"
        )
        y_pred = clf.classes_[np.argmax(y_proba, axis=1)]

        # Plot results
        fig, (ax1, ax2) = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)

        with open(f"./figures/single_food_{food}_classification_report.txt", "w") as f:
            f.write(f"n features = {len(X.columns)}\n")
            f.write(classification_report(y_true, y_pred))
            f.write("\n")

        skplt.metrics.plot_confusion_matrix(
            y_true, y_pred, ax=ax1, title_fontsize=24, text_fontsize=20
        )
        skplt.metrics.plot_roc(
            y_true,
            y_proba,
            ax=ax2,
            title_fontsize=24,
            text_fontsize=12,
            plot_micro=False,
        )

        plt.savefig(
            f"./figures/single_food_{food}_classification_results.png",
            bbox_inches="tight",
        )
        plt.close()

        # Store feature importance per food if we are using random forest
        # extract these from an RF trained on the full data
        feature_importance = {}

        K = len(X.columns)
        # clf was already fit on the full data since we wanted class indices
        best_feature_idxs = np.argsort(clf.feature_importances_)[::-1]
        best_features = list(X.columns[best_feature_idxs[:K]])
        best_features_names = list(
            map(lambda ko: f"{ko}  {get_ko_name(ko)}", best_features)
        )
        best_feature_importances = clf.feature_importances_[best_feature_idxs[:K]]
        best_features_all[food] = list(
            zip(best_features, best_features_names, best_feature_importances)
        )

        best_features_df = pd.DataFrame(best_features_all[food])
        best_features_df.columns = ["KO", "KO Name", "Importance"]
        best_features_df = best_features_df.set_index("KO")

        best_features_df["pathways"] = best_features_df.index.map(
            lambda ko: ";; ".join(get_pathway_names(get_gene_pathways(ko)))
        )
        best_features_df["brite"] = best_features_df.index.map(
            lambda ko: "; ".join(get_gene_brite(ko, levels=["H2", "H3"]))
        )

        best_features_df.to_csv(f"./figures/single_food_{food}_feature_importance.csv")

        imps = sorted(clf.feature_importances_[:50], reverse=True)
        raw_imps.append((food, imps))

    if PLOT_FEATURE_IMPORTANCE:
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.title(f"")
        ax.set_xlabel("Feature Rank", size=20)
        ax.set_ylabel("Relative Feature Importance", size=20)
        styles = ["solid", "dotted", "dashed"]

        for i, (food, imp) in enumerate(raw_imps):
            ax.plot(imp / imp[0], lw=4, color="black", ls=styles[i], label=food)

        plt.legend()
        plt.savefig(
            "./figures/single_food_feature_importances.svg", bbox_inches="tight"
        )
        plt.close()

    return best_features_all


best_features_sf = single_food_classification()


## Multi-food classification
def multi_food_classification():

    FEATURE_IMPORTANCE = True
    PLOT_FEATURE_IMPORTANCE = True
    LFC_FILTER = 2

    kegg_b, kegg_e, metadata_b, metadata_e = load_raw_data()
    kegg_b, kegg_e = log_normalize(kegg_b), log_normalize(kegg_e)
    kegg_d = calculate_difference(kegg_b, kegg_e)

    dge = calc_dge()
    if LFC_FILTER != 0:
        dge = filter_dge_by_lfc(dge, LFC_FILTER)

    kegg_treat = []
    kegg_labels = []
    dges = []

    # Create dataset
    for food in dge.keys():

        if len(dge[food]["genes"]) == 0:
            continue

        # Record treatment set for this food
        kegg_food = kegg_d.filter(like=food, axis=0)
        food_control = kegg_food.filter(like=f"No{food}", axis=0)
        food_treatment = kegg_food.loc[~kegg_food.index.isin(food_control.index), :]

        food_control.index = food_control.index.map(lambda i: i.replace("No", ""))

        diff = food_treatment - food_control

        kegg_treat.append(diff)

        # Record DGE genes
        dges.extend(dge[food]["genes"])

        # Record labels
        kegg_labels.extend([food] * len(food_treatment))

    dges = list(set(dges))
    kegg_treat = pd.concat(kegg_treat)
    kegg_treat = kegg_treat[dges]

    # Perform LOO classification

    X = kegg_treat.copy()
    y_true = kegg_labels.copy()

    clf = RandomForestClassifier(
        n_estimators=2000,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
        class_weight="balanced",
    )
    clf.fit(X, y_true)  # fit so we get class assignments

    y_proba = cross_val_predict(
        clf, X, y_true, cv=LeaveOneOut(), n_jobs=-1, method="predict_proba"
    )
    y_pred = clf.classes_[np.argmax(y_proba, axis=1)]

    # Plot results
    with open(f"./figures/multi_food_classification_report.txt", "w") as f:
        f.write(f"n features = {X.shape[1]}\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\n")

    fig, (ax1, ax2) = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)

    skplt.metrics.plot_confusion_matrix(
        y_true, y_pred, ax=ax1, title_fontsize=24, text_fontsize=20
    )
    skplt.metrics.plot_roc(
        y_true,
        y_proba,
        ax=ax2,
        title_fontsize=24,
        text_fontsize=12,
        plot_micro=False,
    )

    plt.savefig("./figures/multi_food_classification_results.png", bbox_inches="tight")
    plt.close()

    # Return feature importances
    best_features_all = {}
    if FEATURE_IMPORTANCE:
        K = len(X.columns)
        # clf was already fit on the full data since we wanted class indices
        best_feature_idxs = np.argsort(clf.feature_importances_)[::-1]

        best_features = list(X.columns[best_feature_idxs[:K]])
        best_features_names = list(
            map(lambda ko: f"{ko}  {get_ko_name(ko)}", best_features)
        )

        best_feature_importances = clf.feature_importances_[best_feature_idxs[:K]]
        best_features_all = list(
            zip(best_features, best_features_names, best_feature_importances)
        )

        best_features_df = pd.DataFrame(best_features_all)
        best_features_df.columns = ["KO", "KO Name", "Importance"]
        best_features_df = best_features_df.set_index("KO")
        best_features_df["pathways"] = best_features_df.index.map(
            lambda ko: ";; ".join(get_pathway_names(get_gene_pathways(ko)))
        )
        best_features_df["brite"] = best_features_df.index.map(
            lambda ko: "; ".join(get_gene_brite(ko, levels=["H2", "H3"]))
        )
        best_features_df.to_csv(f"./results/multi_food_feature_importance.csv")

        if PLOT_FEATURE_IMPORTANCE:
            imp = sorted(clf.feature_importances_[:50], reverse=True)

            fig, ax = plt.subplots(figsize=(5, 5))
            plt.title(f"")

            ax.set_xlabel("Feature Rank", size=20)
            ax.set_ylabel("Relative Feature Importance", size=20)

            ax.plot(imp / imp[0], lw=4, color="black")

            plt.savefig(
                "./figures/multi_food_feature_importance.svg", bbox_inches="tight"
            )
            plt.close()

    return best_features_all


best_features_mf = multi_food_classification()


# Multi-food top features heatmap
def multi_food_top_feature_heatmap():

    # Load top features data
    K = 25
    foods = ["Almond", "Walnut", "Broccoli"]

    df = pd.read_csv("./results/multi_food_feature_importance.csv").set_index("KO")
    df = df.iloc[:K, :]

    # Load DGE data, include all features
    dge = calc_dge(q=1)

    # Add log2fc for each feature for each food
    for food in foods:
        df[food] = dge[food]["log2fc"][df.index]

    # Plot
    df_plot = df[foods]

    g = clusterMap(df_plot)
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(f"./figures/multi_food_top_feature_heatmap.svg", bbox_inches="tight")
    plt.close()

    # colorbar
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

    cb = mpl.colorbar.ColorbarBase(
        ax,
        orientation="horizontal",
        cmap=sns.diverging_palette(217, 30, s=85, l=60, as_cmap=True),
        norm=mpl.colors.Normalize(vmin=-5, vmax=5),
    )
    cb.set_label(r"$log_2FC$", fontsize=32)
    cb.ax.tick_params(labelsize=32)
    plt.xticks(ticks=[-5, -2.5, 0, 2.5, 5])

    plt.savefig(
        f"./figures/multi_food_top_feature_heatmap_legend.svg", bbox_inches="tight"
    )
    plt.close()

    # Create dataframe for export
    df = df.drop(["Almond", "Broccoli", "Walnut"], axis=1)

    for food in foods:
        # Create gene-->p series
        series = pd.Series(data=dge[food]["p"], index=dge[food]["genes"])

        df[f"{food}-q"] = series[df.index]
        df[f"{food}-log2fc"] = dge[food]["log2fc"][df.index]
        df[f"{food}-treatment-mean-diff"] = dge[food]["treatment-mean-diff"][df.index]
        df[f"{food}-control-mean-diff"] = dge[food]["control-mean-diff"][df.index]

    df.to_csv("./results/multi_food_top_feature_info.csv")


multi_food_top_feature_heatmap()
