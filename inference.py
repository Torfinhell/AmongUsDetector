from models.fcos_pretrained import ModelFcosPretrained
from cyclopts import App
from src.configs import ModelPredConfig
from src.utils import set_seed
from src.data_module import AmongUsDatamodule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

app = App(name="Define Config for inferencing:")


@app.command
def run_inference(cfg: ModelPredConfig):
    """
    Run inference on all images in a directory and save results with bounding boxes

    Args:
        image_dir: directory containing images to process
        output_dir: directory to save output images with bounding boxes
        checkpoint_path: path to the model checkpoint file
        confidence_threshold: confidence threshold for drawing boxes
    """
    inference_cfg = cfg.inference_cfg
    set_seed(inference_cfg.seed)

    # initialize Datamodule
    data_module = AmongUsDatamodule(cfg.datamodule_cfg, cfg.creation_cfg)
    # Gradient Norm Output
    trainer = L.Trainer(
        accelerator="gpu",
        enable_progress_bar=True,
    )
    model = ModelFcosPretrained(cfg)
    trainer.predict(model=model, datamodule=data_module)
    return trainer, model


if __name__ == "__main__":
    app()
