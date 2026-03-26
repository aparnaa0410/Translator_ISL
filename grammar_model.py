from transformers import T5Tokenizer, T5ForConditionalGeneration

# 🔥 Load lightweight model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# 🔥 Cache (for speed)
last_gloss = ""
last_output = ""


# 🔥 Step 1: Phrase handling
def merge_phrases(words):
    i = 0
    merged = []

    while i < len(words):
        # TAKE CARE
        if i < len(words) - 1 and words[i] == "TAKE" and words[i+1] == "CARE":
            merged.append("TAKE CARE")
            i += 2

        # THANK YOU
        elif i < len(words) - 1 and words[i] == "THANK" and words[i+1] == "YOU":
            merged.append("THANK YOU")
            i += 2

        else:
            merged.append(words[i])
            i += 1

    return merged


# 🔥 Step 2: Reordering rules
def reorder_gloss(words):
    words = merge_phrases(words)

    # Copy list
    words = words.copy()

    # Rule 1: Move "FRIEND" to end
    if "FRIEND" in words:
        words.remove("FRIEND")
        words.append("FRIEND")

    # Rule 2: WANT structure (I WATER WANT → I WANT WATER)
    if "WANT" in words:
        words.remove("WANT")
        words.insert(1, "WANT")

    # Rule 3: Pronoun positioning
    if "I" in words:
        words.remove("I")
        words.insert(0, "I")

    if "YOU" in words:
        words.remove("YOU")
        words.insert(0, "YOU")

    return words


# 🔥 Step 3: Main function
def convert_gloss_to_sentence(gloss_words):
    global last_gloss, last_output

    if not gloss_words:
        return ""

    # Apply reordering
    reordered_words = reorder_gloss(gloss_words)

    gloss = " ".join(reordered_words)

    # Cache check
    if gloss == last_gloss:
        return last_output

    # 🔥 NLP polishing using T5
    prompt = f"Convert to a natural English sentence: {gloss}"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=30)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Capitalize properly
    if result:
        result = result[0].upper() + result[1:]

    # Save cache
    last_gloss = gloss
    last_output = result

    return result