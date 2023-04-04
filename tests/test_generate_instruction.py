"""
Before Run: mkdir -p /home/app/open && cd /home/app/open && git clone https://github.com/tatsu-lab/stanford_alpaca.git

export DEBUG_PORT=5679
REMOTE_DEBUG=1 python -m tests.test_generate_instruction generate_instruction_following_data  \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003"
"""
import sys
sys.path.insert(0, "/home/app/open/stanford_alpaca")

import fire
from generate_instruction import *


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
