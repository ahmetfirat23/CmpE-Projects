import numpy as np
from homework1 import Hw1Env
from PIL import Image
from multiprocessing import Process
import torch

env = Hw1Env(render_mode="offscreen")
def collect(idx, N):
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    for i in range(N):
        env.reset()
        action_id = np.random.randint(4)
        _, img_before = env.state()
        env.step(action_id)
        _, img_after = env.state()
        actions[i] = action_id
        imgs_before[i] = img_before
        imgs_after[i] = img_after
        env.reset()
    torch.save(actions, f"hw1/data/actions_unet_{idx}.pt")
    torch.save(imgs_before, f"hw1/data/imgs_before_unet_{idx}.pt")
    torch.save(imgs_after, f"hw1/data/imgs_after_unet_{idx}.pt")


if __name__ == "__main__":
    processes = []
    for i in range(10):
        p = Process(target=collect, args=(i, 100))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()