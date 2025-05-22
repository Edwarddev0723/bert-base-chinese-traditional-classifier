from data_utils import prepare_dataset
from src.pipeline import TextClassifierPipeline

pipeline = TextClassifierPipeline()
pipeline.prepare_data(prepare_dataset_func=prepare_dataset)
pipeline.setup_model()
pipeline.train()
y_true, y_pred, cm, report = pipeline.evaluate()
pipeline.visualize_confusion_matrix(cm)
pipeline.error_analysis(y_true, y_pred)
