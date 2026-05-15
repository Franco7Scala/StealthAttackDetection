import torch
import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
from src.arn.model import Generator
from src.prototypical_net.proto_model import ProtoModel
from src.prototypical_net.prototypical_loss import prototypical_loss


def train_proto(args, x_train_unsupervised, x_train_few_shot):
    #model = Generator(nf_in=x_train_unsupervised.shape[1], nf_out=args.nf_out, z_dim=args.z_dim).to(args.device)
    model = ProtoModel(input_dim=x_train_unsupervised.shape[1], output_dim=args.z_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    classes = {0: x_train_unsupervised, 1: torch.tensor(x_train_few_shot, dtype=torch.float32)}
    n_support = 3  # n samples for prototype
    n_query = 5  # n samples for test loss
    model.train()
    pbar = tqdm(range(args.num_epochs), desc="Training Epochs")
    for epoch in pbar:
        optimizer.zero_grad()
        episode_x = []
        episode_y = []
        for c in range(len(classes.keys())):
            class_samples = classes[c]
            idx = torch.randperm(len(class_samples))[:n_support + n_query]
            samples = class_samples[idx]
            episode_x.append(samples.to(args.device))
            episode_y.extend([c] * (n_support + n_query))

        episode_x = torch.cat(episode_x).to(args.device)
        episode_y = torch.tensor(episode_y, dtype=torch.long).to(args.device)

        _, mu, _, _ = model(episode_x)
        loss, acc = prototypical_loss(mu, episode_y, n_support)
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}, Pseudo-Accuracy = {acc.item():.4f}")

    return model
