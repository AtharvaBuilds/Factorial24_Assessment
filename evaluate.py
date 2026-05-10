import pandas as pd
from rfq_engine import RFQEngine
import time
from tabulate import tabulate

def generate_test_cases(engine, num_cases=10):
    # Select random products
    sample_products = engine.bom_df.groupby('product_id').first().reset_index().sample(num_cases, random_state=42)
    
    test_cases = []
    for _, row in sample_products.iterrows():
        # Create variations of RFQ text
        # Example: "Need {qty} {product_type} with {dial_size}mm dial, up to {range_max} {unit}. {connection_size} {connection_type} connection, {mounting} mounting."
        qty = 50
        
        # We introduce some "noise" or different phrasing
        text_variations = [
            f"Please quote for {qty} {row['product_type'].lower()}s. Specs: {row['dial_size_mm']}mm dial, {row['range_max']} {row['unit']}, {row['connection_size']} {row['connection_type']}, {row['mounting']} mount.",
            f"Looking to order {qty} units of {row['product_type']}. It needs a {row['dial_size_mm']}mm dial, range up to {row['range_max']}{row['unit']}. Mounting should be {row['mounting']} with {row['connection_size']} {row['connection_type']} connection.",
            f"We require {qty}x {row['product_type']}, {row['dial_size_mm']}mm, {row['range_max']} {row['unit']}, {row['connection_size']} {row['connection_type']} {row['mounting']}."
        ]
        
        for text in text_variations:
            test_cases.append({
                'rfq_text': text,
                'expected_product_id': row['product_id']
            })
            
    return test_cases

def run_evaluation():
    print("Initializing Engine...")
    start_time = time.time()
    engine = RFQEngine()
    print(f"Engine Initialized in {time.time() - start_time:.2f} seconds.")
    
    print("Generating Test Cases...")
    test_cases = generate_test_cases(engine, num_cases=10) # 30 test cases total
    
    print(f"Running Evaluation on {len(test_cases)} RFQs...\n")
    
    correct_matches = 0
    results = []
    
    for i, test in enumerate(test_cases):
        rfq_text = test['rfq_text']
        expected_id = test['expected_product_id']
        
        report = engine.process_rfq(rfq_text)
        matched_id = report['matched_product_id']
        confidence = report['match_confidence']
        
        is_correct = (matched_id == expected_id)
        if is_correct:
            correct_matches += 1
            
        results.append([
            i+1,
            rfq_text[:40] + "...",
            expected_id,
            matched_id,
            f"{confidence:.2f}",
            "[Pass]" if is_correct else "[Fail]"
        ])
    
    accuracy = (correct_matches / len(test_cases)) * 100
    
    print(tabulate(results, headers=["#", "RFQ Text (Truncated)", "Expected ID", "Matched ID", "Confidence", "Result"], tablefmt="grid"))
    
    print("\n--- Evaluation Metrics ---")
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Correct Matches: {correct_matches}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("--------------------------\n")

if __name__ == '__main__':
    run_evaluation()
