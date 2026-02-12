import lightning as L
from cyclopts import App
from lightning.pytorch import seed_everything

from src.configs import ModelPredConfig
from src.data_module import AmongUsDatamodule
from src.models.fcos_pretrained import ModelFcosPretrained
from src.utils import create_output

app = App(name="Define Config for inferencing:")


@app.command
def run_inference(cfg: ModelPredConfig = ModelPredConfig()):
    """
    Run inference on all images in a directory and save results with bounding boxes

    Args:
        image_dir: directory containing images to process
        output_dir: directory to save output images with bounding boxes
        checkpoint_path: path to the model checkpoint file
        confidence_threshold: confidence threshold for drawing boxes
    """
    inference_cfg = cfg.inference_cfg
    seed_everything(inference_cfg.seed)

    # initialize Datamodule
    data_module = AmongUsDatamodule(
        cfg.datamodule_cfg, cfg.creation_cfg, cfg.transform_cfg
    )
    # Gradient Norm Output
    trainer = L.Trainer(
        accelerator="gpu",
        enable_progress_bar=True,
    )
    model = ModelFcosPretrained.load_from_checkpoint(
        inference_cfg.checkpoint, weights_only=False
    )
    output = trainer.predict(model=model, datamodule=data_module)
    images_paths, preds = [item[0] for item in output], [item[1] for item in output]
    create_output(
        images_paths,
        preds,
        (cfg.transform_cfg.width, cfg.transform_cfg.height),
        cfg.datamodule_cfg.pred_output,
    )


if __name__ == "__main__":
    app()
