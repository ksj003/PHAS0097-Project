import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black", "xtick.bottom": True, "ytick.left": True})

# Define file paths for inputs and outputs
INPUT_FILE = '/mnt/data2/student/Karam/docking_validation/results_csv/all_results.csv'
OUTPUT_DIR = '/mnt/data2/student/Karam/docking_validation/plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Ensure key metrics are numeric; invalid entries become NaN
numeric_cols = ['RMSD', 'IFP_Tanimoto', 'Contact_Recovery_Pct', 'Burial_Similarity', 'Pocket_CA_RMSD', 'Validity_Pct']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Scale fraction-based metrics to percentages for easier interpretation
df['IFP_Tanimoto'] = df['IFP_Tanimoto'] * 100
df['Burial_Similarity'] = df['Burial_Similarity'] * 100

# Order phases chronologically to ensure consistent plotting on the x-axis
phase_order = ['P1_Redock', 'P2_Crossdock', 'P3_Apo']
df['Phase'] = pd.Categorical(df['Phase'], categories=phase_order, ordered=True)

# Define standard metrics and their corresponding axis labels
metrics_list_summary = ['RMSD', 'IFP_Tanimoto', 'Contact_Recovery_Pct', 'Burial_Similarity']
metric_titles_list = ['Average RMSD', 'Average IFP Tanimoto', 'Average Contact Recovery', 'Average Burial Similarity']
metric_ylabels = ['RMSD (Å)', 'IFP Tanimoto (%)', 'Contact Recovery (%)', 'Burial Similarity (%)']

# Save the figure with tight bounds and free up memory
def save_and_close(fig, filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

# Figure 4: Plot pocket backbone RMSD stability comparing Boltz and Chai-1
def plot_figure_4():
    df_f4 = df[df['Tool'].isin(['Boltz', 'Chai-1'])].dropna(subset=['Pocket_CA_RMSD'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(data=df_f4, x='Tool', y='Pocket_CA_RMSD', color='black', alpha=0.5, jitter=True, ax=ax, zorder=1)
    sns.boxplot(data=df_f4, x='Tool', y='Pocket_CA_RMSD', palette='Set2', ax=ax, 
                whis=(0, 100), showfliers=False, boxprops={'alpha': 0.8}, zorder=2)
    
    ax.set_title('Figure 4: Pocket Backbone RMSD Stability (Boltz vs Chai-1)')
    ax.set_ylabel('Pocket RMSD (Å)') 
    
    save_and_close(fig, 'Figure_4_Pocket_RMSD_Stability.png')

# Globally exclude Chai-1 from all subsequent analyses
df_filtered = df[df['Tool'] != 'Chai-1'].copy()

# Figure 5: Bar chart showing the overall pose validity percentage for each tool
def plot_figure_5():
    num_tools = len(df_filtered['Tool'].unique())
    tool_palette = sns.color_palette("plasma", num_tools)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_filtered, x='Tool', y='Validity_Pct', palette=tool_palette, ax=ax)
    
    ax.set_title('Figure 5: Overall Generative Pose Validity by Tool')
    ax.set_ylabel('Mean Validity (%)')
    ax.set_xlabel('Docking Tool')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    save_and_close(fig, 'Figure_5_Overall_Validity.png')

# Generate a 2x2 subplot summarising key metrics for a specific docking phase
def plot_phase_summary_metrics(phase_name, filename, title_prefix):
    df_phase = df_filtered[df_filtered['Phase'] == phase_name]
    grouped_phase = df_phase.groupby('Tool')[metrics_list_summary].mean().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix} Performance Summary by Tool', fontsize=16)
    
    axes_flat = axes.flatten()
    for i, metric in enumerate(metrics_list_summary):
        sns.barplot(data=grouped_phase, x='Tool', y=metric, ax=axes_flat[i], palette='viridis')
        axes_flat[i].set_title(metric_titles_list[i])
        axes_flat[i].set_xticklabels(axes_flat[i].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        axes_flat[i].set_xlabel('')
        axes_flat[i].set_ylabel(metric_ylabels[i])
        
    save_and_close(fig, filename)

# Figure 6: Phase 1 baseline self-docking summary
def plot_figure_6():
    plot_phase_summary_metrics('P1_Redock', 'Figure_6_Phase1_Baseline.png', 'Figure 6: Phase 1 (Baseline Self-Docking)')

# Figure 7: Phase 2 cross-docking summary
def plot_figure_7():
    plot_phase_summary_metrics('P2_Crossdock', 'Figure_7_Phase2_Summary.png', 'Figure 7: Phase 2 (Cross-Docking)')

# Figure 8: Line plot tracking IFP Tanimoto degradation across the three docking phases
def plot_figure_8():
    grouped_ift = df_filtered.groupby(['Phase', 'Tool'])['IFP_Tanimoto'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=grouped_ift, x='Phase', y='IFP_Tanimoto', hue='Tool', marker='s', ax=ax, linewidth=2, markersize=8)
    
    ax.set_title('Figure 8: Performance Degradation Across Phases', pad=15)
    ax.set_ylabel('Mean IFP Tanimoto Similarity (%)')
    ax.set_xlabel('Docking Phase')
    ax.legend(title='Docking Tool', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_8_IFT_Phase_Degradation.png'), dpi=300)
    plt.close()

# Figure 9: Phase 3 apo-docking summary
def plot_figure_9():
    plot_phase_summary_metrics('P3_Apo', 'Figure_9_Phase3_Summary.png', 'Figure 9: Phase 3 (Apo-Docking)')

# Figure 11: Heatmap detailing Phase 3 IFP Tanimoto scores by target mutation and tool
def plot_figure_11():
    df_f11 = df_filtered[df_filtered['Phase'] == 'P3_Apo'].copy()
    
    pivot_df_f11 = df_f11.pivot_table(index=['Mutation', 'Lig_PDB'], columns='Tool', values='IFP_Tanimoto', aggfunc='mean')
    pivot_df_f11 = pivot_df_f11.sort_index(level='Mutation')
    heatmap_data_f11 = pivot_df_f11.reset_index(level='Mutation').drop(columns='Mutation')
    
    mutations_f11 = pivot_df_f11.index.get_level_values('Mutation')
    split_idx_f11 = list(mutations_f11).index('G13D') if 'G13D' in mutations_f11 else None
    
    fig, ax = plt.subplots(figsize=(14, 10)) 
    
    # Format the heatmap with thin white cell borders
    sns.heatmap(heatmap_data_f11, annot=True, cmap='YlGnBu', vmin=0, vmax=100, fmt=".1f",
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Mean IFP Tanimoto (%)'}, ax=ax)
                
    # Disable the standard background grid to prevent overlapping lines
    ax.grid(False)
    
    # separate G12D and G13D targets with a thick white line if both exist in the data
    if split_idx_f11:
        ax.axhline(split_idx_f11, color='white', lw=6)
        ax.text(-0.5, split_idx_f11 / 2, 'G12D Targets', va='center', ha='right', fontsize=12, fontweight='bold')
        ax.text(-0.5, split_idx_f11 + (len(mutations_f11) - split_idx_f11) / 2, 'G13D Targets', va='center', ha='right', fontsize=12, fontweight='bold')

    ax.set_title('Figure 11: Phase 3 IFP Tanimoto Heatmap (By Target and Tool)', pad=20)
    ax.set_ylabel('Target Ligand')
    ax.set_xlabel('') 
    
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.2) 
    plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_11_Phase3_Heatmap.png'), dpi=300)
    plt.close()

# Figure 12: Bar chart comparing the average computational runtime per pose
def plot_figure_12():
    runtime_data = {
        'Tool': ['DiffDock-P', 'DiffDock', 'Vina', 'Smina', 'Boltz', 'Boltz+DD', 'Boltz+Vina'],
        'Time_Seconds': [4, 5, 33, 33, 108, 112, 140]
    }
    df_rt = pd.DataFrame(runtime_data)
    df_rt = df_rt.sort_values('Time_Seconds')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_rt, x='Tool', y='Time_Seconds', palette='crest', ax=ax)
    
    ax.set_title('Figure 12: Average Computational Runtime per Pose by Tool')
    ax.set_ylabel('Average Runtime (Seconds)')
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    
    # Annotate each bar with the exact runtime in seconds
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}s", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', xytext=(0, 5), 
                    textcoords='offset points', fontsize=10, fontweight='bold')
                    
    save_and_close(fig, 'Figure_12_Tool_Runtimes.png')

plot_figure_4()
plot_figure_5()
plot_figure_6()
plot_figure_7()
plot_figure_8()
plot_figure_9()
plot_figure_11()
plot_figure_12()