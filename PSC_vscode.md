# General Commands and Instructions for connecting to PSC

Adapted from this link: [Reference](https://gist.github.com/clane9/2e843b708ab77f4f0526b3bf57268adb)

## 2. Start Interactive Session

Next, start an [interactive session](https://www.psc.edu/resources/bridges-2/user-guide-2-2/#interactive-sessions). I like to `ssh bridges2`, then create a [screen](https://linuxize.com/post/how-to-use-linux-screen/) session (in case my connection drops)

```sh
# Start or resume a screen session named "A"
screen -dR A
```

Then launch an interactive session

```sh
# Start an interactive session with 8 CPUs for 2 hours
interact -n 8 -t 2:00:00
```

```sh
# Start an interactive session with gpu node with one gpu for 2hrs
interact -p GPU-shared --gres=gpu:1 -t 2:00:00
```

Once the interactive session starts, make a note of what node you were assigned. Then you can just sleep inside the terminal (so the scheduler doesn't automatically cancel the job) and minimize the terminal--we won't need it anymore.

```sh
# Sleep for 30 secs on repeat
watch sleep 30
```

## 3. Start a remote VS code session on the compute node

Now you just need to open a [remote VS Code session](https://code.visualstudio.com/docs/remote/ssh) to the compute node. Open VS Code, click the "Open a Remote window" button in the lower left, select "Connect to Host". You should hopefully see the compute node where your interactive session is running in the dropdown (if not, return to [Step 1](#1-create-an-ssh-route-to-the-compute-node)). Once the session starts, you can open your project as usual.

* Note: youo might have too add the node to your ssh config file as a new host.


## 4. Closing the interactive session (optional)

When you're done, you just close everything in reverse

- Close the VS Code window
- Exit the interactive session running in the separate terminal
- Exit the screen session

If you forget, the interactive session will just run to completion, and then SLURM will kill everything for you.


## Setup Virtual Environment

```sh
module load anaconda3/2024.10-1

# Load cuda
module load cuda/12.4.0

# Step1: create an environment
conda create -n env_name python=3.9.16

# Step2: activate the environment
conda activate env_name

# Step3: install the requirements
cd [PROJECT DIRECTORY]
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements.extra.txt
pip install -Ue .
```

## Sending initial project too the node over SCP if needed

```sh
scp -r llmsys_f25_hw5 rhamor@data.bridges2.psc.edu:~/projects/
```

## Handling Disk Quota Limit

PSC sets strict 10gb limit for users home folder so user should be mindful and make sure to stay under the limit. If you go to your home folder and use the command

```sh
du -sh
```

You will be able to see the current disk usage. Here are some steps to help keep the home folder clean.

1. Clean pip cache

```sh
pip cache purge
```

2. Remove old enviornments

```sh
conda deactivate
conda env remove --name <enviornment_name>
```

