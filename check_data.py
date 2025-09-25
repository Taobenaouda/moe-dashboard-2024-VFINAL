#!/usr/bin/env python3
"""
Script de vérification des données pour le dashboard MOE
"""
import os
import json
import pandas as pd

def check_required_files():
    """Vérifie que tous les fichiers requis sont présents"""
    required_files = [
        "out/kpis.json",
        "out/registrations_slim.csv",
        "dashboard_app.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Fichiers manquants:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ Tous les fichiers requis sont présents")
        return True

def check_data_integrity():
    """Vérifie l'intégrité des données"""
    try:
        # Vérifier le JSON KPIs
        with open("out/kpis.json", "r", encoding="utf-8") as f:
            kpis = json.load(f)
        print(f"✅ KPIs chargés: {len(kpis)} entrées")
        
        # Vérifier le CSV principal
        df = pd.read_csv("out/registrations_slim.csv", dtype=str)
        print(f"✅ Dataset principal: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Vérifier les colonnes essentielles
        required_columns = ["REF", "parcours_norm", "sexe", "age", "present"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Colonnes manquantes: {missing_cols}")
            return False
        else:
            print("✅ Toutes les colonnes essentielles sont présentes")
            return True
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification des données: {e}")
        return False

def main():
    print("🔍 Vérification du projet MOE Dashboard...")
    print("=" * 50)
    
    files_ok = check_required_files()
    data_ok = check_data_integrity()
    
    print("=" * 50)
    if files_ok and data_ok:
        print("🚀 Projet prêt pour le déploiement!")
        return True
    else:
        print("❌ Projet non prêt - corriger les erreurs ci-dessus")
        return False

if __name__ == "__main__":
    main()
