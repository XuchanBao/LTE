import os
from pexpect import pxssh
import getpass

from src.utils.misc import sendline_and_get_response

HOSTNAME = "vremote.vectorinstitute.ai"
USERNAME = "anilcem"
GOTO_ROOT_COMMAND = "goto_audiocaps"
RESULTS_DIR = "results"


def connect_to_server(hostname=HOSTNAME, username=USERNAME, port=None):
    try:
        s = pxssh.pxssh()

        # ____ Connect. ____
        print("Connecting. ")
        password = getpass.getpass('password: ')
        s.login(hostname, username, password, port=port)
        print("Done. ")
        return s
    except pxssh.ExceptionPxssh as e:
        print(e)


def generate_experiments_on_server(s, exp_dir, goto_root_command=GOTO_ROOT_COMMAND, results_dir=RESULTS_DIR):
    try:
        # ____ Generate experiments. ____
        # Move the project, activate conda environment.
        print("\n Setting up project. \n")
        sendline_and_get_response(s, "source ~/.bashrc")
        sendline_and_get_response(s, goto_root_command)
        sendline_and_get_response(s, "conda activate audiocaps")

        # Generate the experiments.
        print("\n Generating experiments. \n")
        sendline_and_get_response(s, "bash src/mains/generate_experiments.sh {}".format(exp_dir))

        # Check the generated experiments.
        print("\n Verifying generated experiments. \n")
        sendline_and_get_response(s, "find {} -type f -name *.yaml".format(os.path.join(exp_dir, results_dir)))

        # ____ Done. Log out. ____
        print("Done")
    except pxssh.ExceptionPxssh as e:
        print(e)


def run_experiments_on_server(s, exp_dir, goto_root_command=GOTO_ROOT_COMMAND):
    try:
        # ____ Run experiments. ____
        # Move the project, activate conda environment.
        print("\n Setting up project. \n")
        sendline_and_get_response(s, "source .bashrc")
        sendline_and_get_response(s, goto_root_command)
        sendline_and_get_response(s, "conda activate audiocaps")

        # Run experiments.
        print("\n Running experiments. \n")
        sendline_and_get_response(s, "cd {}".format(exp_dir))
        sendline_and_get_response(s, "sbatch batch_run.sh")

        # ____ Done. Log out. ____
        print("Done")
    except pxssh.ExceptionPxssh as e:
        print(e)
