import re
import ast
import numpy as np
import matplotlib.pyplot as plt


def extract_metrics_from_log(log_path: str) -> dict:
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = {}

    for line in lines:
        if "multiple_snr" in line and "channel_number" in line:
            # extract lists using regex
            snr_match = re.search(r"'multiple_snr':\s*(\[[^\]]+\])", line)
            ch_match = re.search(r"'channel_number':\s*(\[[^\]]+\])", line)
            if snr_match:
                result["multiple_snr"] = ast.literal_eval(snr_match.group(1))
            if ch_match:
                result["channel_number"] = ast.literal_eval(ch_match.group(1))
            break  # first config line only

    start_indices = [i for i, line in enumerate(lines) if "Start Test:" in line]
    if not start_indices:
        raise ValueError("No test block found.")
    start_idx = start_indices[-1]

    key_map = {
        "SNR (denoised)": "snr_denoised",
        "SNR": "snr",
        "CBR": "cbr",
        "PSNR": "psnr",
        "MS-SSIM": "ms_ssim",
        "SSIM": "ssim",
    }

    for line in lines[start_idx:]:
        if "Finish Test!" in line:
            break
        if "]" not in line:
            continue
        content = line.split("]", 1)[1].strip()
        if ":" not in content:
            continue
        name, value_str = content.split(":", 1)
        name = name.strip()
        if name in key_map:
            result[key_map[name]] = ast.literal_eval(value_str.strip())

    return result


def plot_lines(x, y, z, xlabel="x", ylabel="y", zlabel="z"):
    """
    x: 1D array of shape (N,)
    y: 1D array of shape (M,)
    z: 2D array of shape (M, N)
       each row corresponds to one y setting
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    assert z.shape == (
        len(y),
        len(x),
    ), f"z shape must be ({len(y)}, {len(x)}) but got {z.shape}"

    plt.figure()

    for i, y_val in enumerate(y):
        plt.plot(x, z[i], label=f"{ylabel} = {y_val}")

    plt.xlabel(xlabel)
    plt.ylabel(zlabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_path = "./logs/test.log"
    result = extract_metrics_from_log(
        log_path,
    )
    plot_lines(
        result["multiple_snr"],
        result["channel_number"],
        result["psnr"],
        "SNR",
        "CBR",
        "PSNR",
    )
