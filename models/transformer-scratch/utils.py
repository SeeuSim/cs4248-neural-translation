def pad_or_truncate(tokenized_text, pad_idx, max_len=None):
    if len(tokenized_text) < max_len:
        left = max_len - len(tokenized_text)
        padding = [pad_idx] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:max_len]

    return tokenized_text 
