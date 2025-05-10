#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korrelationsanalyse für Nutzerumfrage-Service-Chatbots im Seminararbeit-Stil.
Untersucht Spearman-Rho zwischen:
 - Nutzungsfrequenz (A202) und Zufriedenheit (A401)
 - Wichtigkeit einzelner Merkmale (A301_01–A301_08) und Zufriedenheit (A401)
Erstellt Korrelationsmatrix-CSV und stilisierte Heatmap im einheitlichen Diagramm-Layout.
"""
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_settings  # zentrale RC-Params und PGF-Backend laden


def read_codebook(path, sep='\t'):
    """
    Lese CSV mit möglichen Encodings für LABELs.
    """
    for enc in ['utf-8-sig', 'utf-16', 'ISO-8859-1', 'utf-8']:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, engine='python')
        except Exception:
            continue
    raise Exception(f"Konnte Codebook {path} nicht einlesen.")


def main():
    # Basisverzeichnisse
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'data_chatbot_quality_2025-05-09.xlsx')
    vars_file = os.path.join(base_dir, 'data', 'variables_chatbot_quality_2025-05-09.csv')
    out_dir = os.path.join(base_dir, 'figures', 'correlation')
    os.makedirs(out_dir, exist_ok=True)

    # Daten einlesen und Label-Zeile entfernen
    raw = pd.read_excel(data_file, engine='openpyxl')
    df = raw.iloc[1:].reset_index(drop=True)

    # Numerische Umwandlung
    df['A202'] = pd.to_numeric(df['A202'], errors='coerce')
    df['A401'] = pd.to_numeric(df['A401'], errors='coerce')
    a301_vars = [f'A301_{i:02d}' for i in range(1, 9)]
    for v in a301_vars:
        df[v] = pd.to_numeric(df[v], errors='coerce')

    # 1) Spearman A202 vs A401
    df_pair = df[['A202', 'A401']].dropna()
    rho_pair = df_pair.corr(method='spearman').loc['A202', 'A401']
    with open(os.path.join(out_dir, 'corr_A202_A401.txt'), 'w') as f:
        f.write(f"Spearman-Rho A202 vs A401: {rho_pair:.3f}\n")
    print(f"Spearman-Rho A202↔A401 = {rho_pair:.3f}")

    # 2) Spearman Matrix A301_* vs A401
    subset = df[a301_vars + ['A401']].dropna()
    corr_mat = subset.corr(method='spearman')

    # CSV speichern
    corr_csv = os.path.join(out_dir, 'corr_matrix_A301_A401.csv')
    corr_mat.to_csv(corr_csv, encoding='utf-8-sig')
    print(f"Korrelationsmatrix CSV: {corr_csv}")

    # Heatmap
    # Kurz-Labels für Achsen
    code_to_label = {
        'A301_01': 'Geschwindigkeit',
        'A301_02': 'Genauigkeit',
        'A301_03': 'Problemlösung',
        'A301_04': 'Menschlichkeit',
        'A301_05': 'Bedienbarkeit',
        'A301_06': 'Kontext',
        'A301_07': 'Tonalität',
        'A301_08': 'Datenschutz',
        'A401':    'Zufriedenheit'
    }
    labels = [code_to_label.get(code, code) for code in corr_mat.columns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Divergierende Farbkarte, symmetrisch um 0
    im = ax.imshow(
        corr_mat.values,
        cmap=plt.get_cmap('RdBu_r'),
        vmin=-1,
        vmax=1
    )
    # Farbskala anpassen
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=mpl.rcParams['xtick.labelsize'])

    # Achsenbeschriftung
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=mpl.rcParams['xtick.labelsize'])
    ax.set_yticklabels(labels, fontsize=mpl.rcParams['ytick.labelsize'])

    # Gitterlinien unter dem Heatmap-Raster
    ax.set_xticks([x - 0.5 for x in range(1, len(labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()
    # Als PDF speichern für Latex-Integration
    heatmap_pdf = os.path.join(out_dir, 'heatmap_A301_A401.pdf')
    fig.savefig(heatmap_pdf, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Heatmap PDF: {heatmap_pdf}")


if __name__ == '__main__':
    main()