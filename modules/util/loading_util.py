def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove "_orig_mod" from key if present
        cleaned_key = key.replace("_orig_mod.", "")
        new_state_dict[cleaned_key] = value
    return new_state_dict