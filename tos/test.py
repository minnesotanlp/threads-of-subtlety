from transformers import AutoTokenizer

def split_text_into_segments(text, max_tokens=512, model_name="bert-base-uncased"):
    # Load the Hugging Face tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split the text into paragraphs
    paragraphs = text.split("\n\n")
    segments = []
    current_segment = ""
    current_token_count = 0

    for paragraph in paragraphs:
        # Tokenize the paragraph and count tokens
        token_count = len(tokenizer.tokenize(paragraph))

        # Check if adding this paragraph exceeds the token limit
        if current_token_count + token_count > max_tokens:
            # Save the current segment and start a new one
            segments.append(current_segment.strip())
            current_segment = paragraph
            current_token_count = token_count
        else:
            # Add the paragraph to the current segment
            if current_segment:
                current_segment += "\n\n"  # Add paragraph separator
            current_segment += paragraph
            current_token_count += token_count

    # Add the last segment if it exists
    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments

# Example usage
if __name__ == "__main__":
    segments = split_text_into_segments(long_text, max_tokens=512, model_name="bert-base-uncased")

    for i, segment in enumerate(segments):
        print(f"Segment {i + 1}:\n{segment}\n{'-' * 50}")