import pandas as pd

try:
    bom_df = pd.read_excel('datasets/historical_bom_dataset.xlsx')
    print("BOM Dataset Head:")
    print(bom_df.head())
    print("\nBOM Dataset Info:")
    print(bom_df.info())
    print("-------------------------------------------------")
except Exception as e:
    print(f"Error reading BOM dataset: {e}")

try:
    inv_df = pd.read_excel('datasets/inventory_dataset.xlsx')
    print("Inventory Dataset Head:")
    print(inv_df.head())
    print("\nInventory Dataset Info:")
    print(inv_df.info())
    print("-------------------------------------------------")
except Exception as e:
    print(f"Error reading Inventory dataset: {e}")
