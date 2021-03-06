import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params


class PredictionEval:
    def __init__(self, map_params=Params()):
        self.map_params = map_params
        self._extend_map_params()
        self.img_name2id = {}
        self.img_id2name = {}

    def _extend_map_params(self):
        params = Params()
        params.iouThrs = np.round(
            np.linspace(0.05, 0.95, int((0.95 - 0.05) / 0.05) + 2, endpoint=True), 3
        )
        self.map_params = params

    def load_data_coco_files(self, annotations_path, predictions_path, train_val_names=None):

        self.cocoGt = COCO(annotations_path)
        for image in self.cocoGt.imgs.values():
            self.img_name2id[image["file_name"]] = image["id"]
            self.img_id2name[image["id"]] = image["file_name"]

        self.cocoDt = self.cocoGt.loadRes(predictions_path)
        self.cocoEval = COCOeval(self.cocoGt, self.cocoDt)
        if train_val_names is not None:
            if train_val_names["type"] == "id":
                self.train_ids = train_val_names["train"]
                self.val_ids = train_val_names["val"]
            elif train_val_names["type"] == "names":
                self.train_ids = [self.img_name2id[name] for name in train_val_names["train"]]
                self.val_ids = [self.img_name2id[name] for name in train_val_names["val"]]

    def default_queries(self):
        queries = [
            {"ap": 1},
            {"ap": 1, "iouThr": 0.1, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.3, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.7, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.9, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "small", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "medium", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "large", "maxDets": 100},
        ]
        return queries

    def get_latex_table(self):
        queries = [
            {"ap": 1},
            {"ap": 1, "iouThr": 0.3, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.75, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "small", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "medium", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "large", "maxDets": 100},
        ]
        val_results = self.evaluate_map(queries, stage="val")
        train_results = self.evaluate_map(queries, stage="train")

        val_results = [round(res, 3) for res in val_results]
        train_results = [round(res, 3) for res in train_results]

        text = f"""stage  & AP & AP@.3 & AP@.5 & AP@.75 & AP@.5_S & AP@.5_M & AP@.5_L \\ \hline
        training & {train_results[0]}& {train_results[1]} & {train_results[2]} & {train_results[3]} 
        & {train_results[4]} & {train_results[5]} & {train_results[6]}
        validation & {val_results[0]}& {val_results[1]} & {val_results[2]} & {val_results[3]} 
        & {val_results[4]} & {val_results[5]} & {val_results[6]}
        """
        return text

    def indices_by_stage(self, stage):
        evaluate_imgs = []
        if stage == "val":
            evaluate_imgs = self.val_ids
        elif stage == "train":
            evaluate_imgs = self.train_ids
        else:  # stage == 'all':
            evaluate_imgs = list(self.img_id2name.keys())
        return evaluate_imgs

    def evaluate_map(self, queries, stage="all", summary=False):
        evaluate_imgs = self.indices_by_stage(stage)

        tmp_map_params = self.map_params
        tmp_map_params.imgIds = evaluate_imgs
        self.cocoEval.params = tmp_map_params
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        results = []
        for query in queries:
            print(self.cocoEval._summarize(**query))
            results.append(self.cocoEval._summarize(**query))
        if summary:
            self.cocoEval.summarize()
        return results

    def precision_by_iou(self, iouThr=0.5, stage="all", areRng="all"):
        evaluate_imgs = self.indices_by_stage(stage)

        area_idx = self.cocoEval.params.area_str2idx(areRng)
        iou_idx = np.where(self.cocoEval.params.iouThrs == iouThr)[0][0]

        self.cocoEval.params.imgIds = evaluate_imgs
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        precisions = self.cocoEval.eval["precision"][iou_idx, :, 0, area_idx, 2]
        return precisions
