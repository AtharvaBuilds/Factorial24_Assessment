import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

class RFQEngine:
    def __init__(self, bom_path='datasets/historical_bom_dataset.xlsx', inv_path='datasets/inventory_dataset.xlsx'):
        self.bom_path = bom_path
        self.inv_path = inv_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_data()
        self.build_catalog()

    def load_data(self):
        try:
            self.bom_df = pd.read_excel(self.bom_path)
            # Ensure proper types
            self.bom_df['dial_size_mm'] = self.bom_df['dial_size_mm'].fillna(0).astype(float)
            self.bom_df['range_max'] = self.bom_df['range_max'].fillna(0).astype(float)
            
            self.inv_df = pd.read_excel(self.inv_path)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            self.bom_df = pd.DataFrame()
            self.inv_df = pd.DataFrame()

    def build_catalog(self):
        if self.bom_df.empty:
            self.catalog = []
            return

        # Group by product_id to extract unique properties
        products = self.bom_df.groupby('product_id').first().reset_index()
        
        self.catalog = []
        self.product_descriptions = []
        self.product_ids = []

        for _, row in products.iterrows():
            desc_parts = [f"{row['product_type']}"]
            if row['dial_size_mm'] > 0:
                desc_parts.append(f"{row['dial_size_mm']}mm dial")
            if row['range_max'] > 0:
                desc_parts.append(f"max range {row['range_max']} {row['unit']}")
            if pd.notna(row['connection_size']) and pd.notna(row['connection_type']):
                desc_parts.append(f"{row['connection_size']} {row['connection_type']} connection")
            if pd.notna(row['mounting']):
                desc_parts.append(f"{row['mounting']} mounting")
                
            description = ", ".join(desc_parts)
            self.catalog.append({
                'product_id': row['product_id'],
                'type': row['product_type'],
                'description': description
            })
            self.product_descriptions.append(description)
            self.product_ids.append(row['product_id'])

        # Precompute embeddings for the catalog
        print("Precomputing product embeddings...")
        self.product_embeddings = self.model.encode(self.product_descriptions)

    def match_rfq(self, rfq_text):
        if not self.catalog:
            return None, 0.0, None

        rfq_embedding = self.model.encode([rfq_text])
        similarities = cosine_similarity(rfq_embedding, self.product_embeddings)[0]
        
        # Get top 5 indices
        top_indices = np.argsort(similarities)[-5:][::-1]
        
        # Extract features from RFQ using regex
        dial_match = re.search(r'(\d+(?:\.\d+)?)\s*mm', rfq_text)
        rfq_dial = float(dial_match.group(1)) if dial_match else None
        
        range_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:psi|bar|C)', rfq_text)
        rfq_range = float(range_match.group(1)) if range_match else None
        
        best_match_idx = top_indices[0]
        best_match_score = similarities[best_match_idx]
        
        # Re-ranking logic
        for idx in top_indices:
            score = similarities[idx]
            desc = self.product_descriptions[idx]
            
            # Boost score if numbers match exact features in the description
            boost = 0.0
            if rfq_dial is not None and (f"{rfq_dial}mm" in desc or f"{int(rfq_dial)}mm" in desc or f"{rfq_dial:.1f}mm" in desc):
                boost += 0.15
            if rfq_range is not None and (f"max range {rfq_range}" in desc or f"max range {int(rfq_range)}" in desc or f"max range {rfq_range:.1f}" in desc):
                boost += 0.15
                
            total_score = score + boost
            if total_score > best_match_score:
                best_match_score = total_score
                best_match_idx = idx

        # Cap score at 0.99 if boosted
        final_score = min(best_match_score, 0.99)
        best_product_id = self.product_ids[best_match_idx]
        best_product_desc = self.product_descriptions[best_match_idx]

        return best_product_id, final_score, best_product_desc

    def generate_bom(self, product_id):
        product_bom = self.bom_df[self.bom_df['product_id'] == product_id]
        bom = []
        for _, row in product_bom.iterrows():
            bom.append({
                'component': row['component'],
                'qty_per_unit': row['qty_per_unit']
            })
        return bom

    def check_inventory(self, bom, requested_qty):
        inventory_dict = self.inv_df.set_index('component')['available_qty'].to_dict()
        
        shortages = []
        for item in bom:
            component = item['component']
            total_required = item['qty_per_unit'] * requested_qty
            available = inventory_dict.get(component, 0)
            
            if total_required > available:
                shortages.append({
                    'component': component,
                    'required': total_required,
                    'available': available,
                    'shortage': total_required - available
                })
                
        return shortages

    def extract_quantity(self, rfq_text):
        # A simple regex to find the first number in the text as a fallback
        # In a real scenario, an LLM or more robust NLP could be used.
        match = re.search(r'\b(\d+)\b', rfq_text)
        if match:
            return int(match.group(1))
        return 1 # default to 1

    def process_rfq(self, rfq_text, requested_qty=None):
        if requested_qty is None:
            requested_qty = self.extract_quantity(rfq_text)

        # 1. Match RFQ to product
        product_id, score, desc = self.match_rfq(rfq_text)
        
        if product_id is None:
            return {"error": "Failed to match product."}

        # 2. Generate BOM
        bom = self.generate_bom(product_id)
        
        # 3. Check Inventory
        shortages = self.check_inventory(bom, requested_qty)
        
        # 4. Feasibility Report
        feasibility = "Feasible" if not shortages else "Not Feasible - Inventory Shortage"
        
        report = {
            "rfq_text": rfq_text,
            "requested_qty": requested_qty,
            "matched_product_id": product_id,
            "matched_description": desc,
            "match_confidence": float(score),
            "feasibility_status": feasibility,
            "bom": bom,
            "shortages": shortages
        }
        
        return report

if __name__ == '__main__':
    engine = RFQEngine()
    test_rfq = "I need 50 pressure gauges with a 25mm dial and back mounting."
    print("\nProcessing RFQ:", test_rfq)
    report = engine.process_rfq(test_rfq)
    
    import json
    print("\nFeasibility Report:")
    print(json.dumps(report, indent=2))
