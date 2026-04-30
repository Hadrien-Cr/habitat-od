import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator

class InstanceWiseEvaluation(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
        """

        self._cpu_device = torch.device("cpu")
        self._metadata = MetadataCatalog.get(dataset_name)
        self.count = 0
        self.num_images = 0

    def reset(self):
        self.count = 0
        self.num_images = 0

    def process(self, inputs, outputs):
        dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            target_class = input["file_name"].split("_habitat_scene_")[0].replace("cls_")
            instances = output["instances"].to(self._cpu_device)
            
            for x in instances:
                raise NotImplementedError
            
            self.num_images += 1


    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if len(self._predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self.count = 0
        self.num_images = 0
        