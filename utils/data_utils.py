def get_batch_data(data, config):
    input, valid = None, None
    if config.dataset_type == "RandomResizedCrop":
        input = data.to(config.device)
    elif config.dataset_type == "LetterBox":
        input = data[0].to(config.device)
        valid = data[1].to(config.device)
    return input, valid
