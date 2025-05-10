#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison Analysis: Top-2-Box für A301 vs. Prozent der Nutzer für A302 und A305.
Normiert A301 (Top-2) und berechnet A302/A305-Prozente pro Nutzer.
Exportiert CSV und PDF-Plot mit zentral definierten Plot-Settings.
"""
import os
import sys

# sicherstellen, dass das Skript-Verzeichnis im Import-Pfad ist
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_settings  # zentrale RC-Params und PGF-Backend laden


def read_tab_csv(path):
    """
    Lese eine Tab-separierte CSV mit möglichen Encodings.
    """
    for enc in ['utf-8-sig', 'utf-16', 'ISO-8859-1']:
        try:
            return pd.read_csv(path, sep='\t', encoding=enc, engine='python')
        except Exception:
            continue
    raise Exception(f"Konnte {path} nicht einlesen.")


def normalize_label(opt):
    """
    Entferne Gruppen-Präfix bis ": " und gebe den Kern zurück.
    """
    if isinstance(opt, str) and ': ' in opt:
        return opt.split(': ', 1)[1]
    return opt


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base, 'data', 'data_chatbot_quality_2025-05-09.xlsx')
    vars_file = os.path.join(base, 'data', 'variables_chatbot_quality_2025-05-09.csv')
    out_dir = os.path.join(base, 'figures', 'comparison')
    os.makedirs(out_dir, exist_ok=True)

    # Rohdaten und Nutzer-Subset
    raw = pd.read_excel(data_file, engine='openpyxl')
    df = raw.iloc[1:].reset_index(drop=True)
    df['A101'] = pd.to_numeric(df['A101'], errors='coerce')
    users = df[df['A101'] == 1].copy()
    users_count = len(users)

    # Codebook/Labels
    variables = read_tab_csv(vars_file)

    # Kürzere Merkmal-Namen (Best-Practice-Begriffe)
    short_labels = {
        'Schnelligkeit der Antworten': 'Geschwindigkeit',
        'Genauigkeit der Antworten': 'Genauigkeit',
        'Fähigkeit, Probleme effektiv zu lösen': 'Problemlösung',
        'Natürliche, menschliche Interaktion': 'Menschlichkeit',
        'Einfache Bedienung (Auswahloptionen)': 'Bedienbarkeit',
        'Kontextverständnis (Chatverlauf wird genutzt)': 'Kontext',
        'Freundlichkeit / Tonalität der Antwort': 'Tonalität',
        'Datenschutz & Vertrauenswürdigkeit': 'Datenschutz'
    }

    # 1) Top-2-Box A301
    a301_vars = [f'A301_{i:02d}' for i in range(1, 9)]
    recs = []
    for var in a301_vars:
        if var not in users:
            continue
        vals = pd.to_numeric(users[var], errors='coerce').dropna()
        pct_top2 = vals.isin([1, 2]).sum() / len(vals) * 100 if len(vals) > 0 else 0
        lbl = variables.loc[variables['VAR'] == var, 'LABEL'].values
        core = normalize_label(lbl[0] if len(lbl) > 0 else var)
        recs.append({'Merkmal': core, 'Top2_Wichtigkeit': round(pct_top2, 2)})
    df_a301 = pd.DataFrame(recs)
    # Long → Short umwandeln, damit es zu A302/A305 passt
    df_a301['Merkmal'] = df_a301['Merkmal'] \
        .map(short_labels) \
        .fillna(df_a301['Merkmal'])

    # 2) A302 (Frage 5): Prozent der Nutzer, die ausgewählt haben
    path302 = os.path.join(base, 'figures', 'descriptives', 'A302_group.csv')
    df_a302 = pd.read_csv(path302, encoding='utf-8-sig')
    df_a302['Frage5'] = df_a302['Häufigkeit'] / users_count * 100
    df_a302['Merkmal'] = df_a302['Option'].apply(normalize_label)
    df_a302 = df_a302[['Merkmal', 'Frage5']]

    # 3) A305 (Frage 6): Prozent der Nutzer, die ausgewählt haben
    path305 = os.path.join(base, 'figures', 'descriptives', 'A305_group.csv')
    df_a305 = pd.read_csv(path305, encoding='utf-8-sig')
    df_a305['Frage6'] = df_a305['Häufigkeit'] / users_count * 100
    df_a305['Merkmal'] = df_a305['Option'].apply(normalize_label)
    df_a305 = df_a305[['Merkmal', 'Frage6']]

    # Zusammenführen
    df_cmp = (
        df_a301
        .merge(df_a302, on='Merkmal', how='inner')
        .merge(df_a305, on='Merkmal', how='inner')
    )

    if df_cmp.empty:
        print('Fehler: Kein Intersection!')
        return

    # CSV & PDF-Plot
    out_csv = os.path.join(out_dir, 'comparison_A301_302_305.csv')
    df_cmp.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print('CSV:', out_csv)

    df_plot = df_cmp.copy()
    df_plot['Merkmal'] = df_plot['Merkmal'].map(short_labels).fillna(df_plot['Merkmal'])
    df_plot = df_plot.rename(columns={
        'Top2_Wichtigkeit': 'Wichtigkeit einzelner Merkmale',
        'Frage5': 'Soll: Am wichtigsten',
        'Frage6': 'Ist-Erfüllung (aktueller Stand)'
    })

    # Sortierung nach größter Differenz zwischen Soll und Ist-Erfüllung
    df_plot['Diff'] = df_plot['Soll: Am wichtigsten'] - df_plot['Ist-Erfüllung (aktueller Stand)']
    df_plot = df_plot.sort_values('Diff', ascending=False)
    df_plot = df_plot.drop(columns=['Diff'])

    # Plot mit Legende unterhalb
    # Nutze DataFrame mit Merkmal als Index
    df_to_plot = df_plot.set_index('Merkmal')
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot mit Farbpalette und Gitter
    colors = plt.get_cmap('Set2').colors
    df_to_plot.plot(kind='bar', ax=ax, color=colors)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_xlabel('')
    ax.set_xticklabels(df_to_plot.index, rotation=45, ha='right', fontsize=mpl.rcParams['axes.labelsize'])
    # plt.title('Top-2-Wichtigkeit vs. Wunsch vs. Ist-Erfüllung')
    plt.ylabel('Prozent der Nutzer', labelpad=14)
    # Legende unterhalb zentriert in drei Spalten
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False
    )
    plt.tight_layout()

    out_pdf = os.path.join(out_dir, 'comparison_A301_302_305.pdf')
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print('Plot als PDF gespeichert:', out_pdf)


if __name__ == '__main__':
    main()
