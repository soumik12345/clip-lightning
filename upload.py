import wandb

with wandb.init(project="clip-image-retrieval", entity="geekyrakshit", job_type="upload"):
    artifact = wandb.Artifact('flickr-8k', type='dataset')
    artifact.add_dir('data')
    wandb.log_artifact(artifact)
