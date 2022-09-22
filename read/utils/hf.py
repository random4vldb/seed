from pytorch_lightning import Callback
from huggingface_hub import Repository


class HuggingFaceHubCallback(Callback):
    def __init__(self, repo_name, local_dir="temp/huggingface-hub"):
        self.repo_name = repo_name
        self.local_dir = local_dir

    def on_init_end(self, trainer):
        self.repo = Repository(local_dir=self.local_dir, clone_from=self.repo_name)

    def on_fit_end(self, trainer, pl_module):
        with self.repo.commit("Add/Update Model"):
            trainer.save_checkpoint(self.local_dir + "/model.ckpt")

        