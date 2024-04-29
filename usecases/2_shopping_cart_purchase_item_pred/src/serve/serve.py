import os
import pickle
import argparse
import glob

from sihg4sr.utils import load_file, prepare_candidate_filter
from sihg4sr.model import SIHG4SR_Handler
from sihg4sr.dataloader import SIHG4SR_DataLoader

class ServeApp:
    def __init__(self, model_dir, feat_dir, candidate_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"{model_dir} does not exist")
        model_path = None
        for name in glob.glob(f'{model_dir}/model*.pth'): 
            model_path = name
        if model_path is None:
            raise FileNotFoundError(f"model is not found in {model_dir} ")

        print(f"Loading model {model_path}")

        if os.path.exists(feat_dir):
            with open(os.path.join(feat_dir, "meta.pkl"), 'rb') as file:
                metadata = pickle.load(file)

        num_items = metadata['num_items']
        num_unique_features = metadata['num_unique_features']
        batch_size = 256
        topk=100
        finetune=False
        
        candidate_df_path = os.path.join(candidate_dir, "candidate_items.csv")
        if os.path.exists(candidate_df_path):
            logit_difference_batch = prepare_candidate_filter(
                candidate_df=load_file(candidate_df_path),
                num_items=num_items,
                batch_size=batch_size
            )
        else:
            logit_difference_batch = None
            
        model_hparams = {}
        self.model = SIHG4SR_Handler(
            num_items=num_items,
            num_unique_features=num_unique_features,
            num_categories=74,
            model_hparams=model_hparams,
            resumed_model=model_path,
            save_path="",
            topk=topk,
            filter=logit_difference_batch,
            finetune=finetune,
            device="cpu",
        )
        self.time_feature = metadata['time_feature']
        self.feature_list = metadata['feature_list']
        self.num_workers = 4
        self.order = 1
        self.attent_longest_view = False
        self.batch_size = batch_size

    def predict(self, sessions_df):
        test_dataloader = SIHG4SR_DataLoader(
            sessions=sessions_df,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=self.order,
            attent_longest_view=self.attent_longest_view,
            enable_sampler=False,
            time_feature=self.time_feature,
            feature_list=self.feature_list,
            recent_n_month=-1,
            sort=False
        )
        preds = self.model.predict(test_dataloader)
        return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", dest="model_dir", default="output/saved_models", type=str)
    parser.add_argument("--feat_dir", dest="feat_dir", default="feat", type=str)
    parser.add_argument("--candidate_dir", dest="candidate_dir", default="output/processed", type=str)
    args = parser.parse_args()
    
    serve = ServeApp(model_dir=args.model_dir, feat_dir = args.feat_dir, candidate_dir=args.candidate_dir)
    
    #sessions_df = load_file(os.path.join(args.candidate_dir, "test_processed.parquet"))[:1]
    sessions_df = load_file(os.path.join(args.feat_dir, "tmp.parquet"))
    preds = serve.predict(sessions_df)
    print(preds)