import os

os.environ["GROWING_LLM"] = "mock"  # tests never reach for the network, whatever .env.local says
