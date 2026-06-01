from edge_casting.training.train import train_profile

for profile in ["edge_sbc", "micro_edge"]:
    print(train_profile(profile))
