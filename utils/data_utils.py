def get_batch_data(data, config):
    input, valid, hr_input = None, None, None
    if config.dataset_type == "RandomResizedCrop":
        if config.sr:
            input = data[0].to(config.device)
            hr_input = data[1].to(config.device)
        else:
            input = data.to(config.device)
    elif config.dataset_type == "LetterBox":
        if config.sr:
            input = data[0].to(config.device)
            valid = data[1].to(config.device)
            hr_input = data[2].to(config.device)
        else:
            input = data[0].to(config.device)
            valid = data[1].to(config.device)
    return input, valid, hr_input
