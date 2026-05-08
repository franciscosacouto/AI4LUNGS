import medspacy
from medspacy.visualization import visualize_ent

# 1. Load a medical NLP pipeline
# This uses rules and deep learning to find clinical entities
nlp = medspacy.load()


def optimize_clinical_text(raw_note):
    doc = nlp(raw_note)
    
    # Extract specific entities relevant to survival (Smoking, Occupation, Condition)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 2. Contextualize (Check for Negation or History)
    # MedSpaacy's "context" component tells us if a smoker "quit" or "never" smoked
    refined_facts = []
    for ent in doc.ents:
        is_negated = ent._.is_negated
        is_historical = ent._.is_historical
        
        status = "Current"
        if is_negated: status = "No"
        elif is_historical: status = "Past/History of"
        
        refined_facts.append(f"{status} {ent.text} ({ent.label_})")

    # 3. Create a "Dense" prompt for MedImageInsight
    # This transforms messy notes into the high-signal format the model prefers
    optimized_prompt = f"Clinical Assessment: {', '.join(refined_facts)}."
    
    return optimized_prompt



# --- Example Usage ---
raw_clinical_data = "Patient is a 65yo male. History of coal mining for 20 years. Quit smoking in 2010. No signs of asthma."
clean_text = optimize_clinical_text(raw_clinical_data)

print(f"Original: {raw_clinical_data}")
print(f"Optimized for Model: {clean_text}")