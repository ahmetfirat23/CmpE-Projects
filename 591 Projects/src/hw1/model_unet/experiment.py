import subprocess

# Define your arguments for each run
arguments_list = [
    # ["--epoch", "200", "--batch_size", "32", "--lr", "0.001","--grayscale", "--loss", "mse", "--save", "unet_gray_mse_no_scheduler"],
    # ["--epoch", "200", "--batch_size", "32", "--lr", "0.001","--loss", "mse", "--save", "unet_mse_no_scheduler"],
    ["--epoch", "300", "--batch_size", "16", "--lr", "0.001","--grayscale", "--loss", "mse", "--save", "unet_gray_mse_scheduler_batchsize16", "--scheduler"],
    # ["--epoch", "200", "--batch_size", "32", "--lr", "0.001", "--loss", "mse", "--save", "unet_mse_scheduler", "--scheduler"],
    # ["--epoch", "200", "--batch_size", "32", "--lr", "0.001","--grayscale", "--loss", "l1", "--save", "unet_gray_l1_scheduler", "--scheduler"],
]

# Loop through the arguments list and run the script for each set of arguments
for arguments in arguments_list:
    # Construct the command to run the script with the current set of arguments
    command = ["python3", "run_unet.py"] + arguments
    print(f"Command executed: {' '.join(command)}")
    # Execute the command synchronously
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Print the result
    print(result.stdout.decode('utf-8'))
    print(result.stderr.decode('utf-8'))
    
