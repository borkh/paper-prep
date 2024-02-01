from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow
from tqdm import tqdm

from .model_search_base import ModelSearchBase


class ModelAnalysis(ModelSearchBase):
    def __init__(
        self,
        start_path: Path,
        split: str,
    ):
        super().__init__(start_path)
        self.split = split

        self.data = self.create_dataframe()

    def _get_tf_events_files(self, version_dir):
        tf_events_files = []
        for tfevent in version_dir.glob("*tfevents*"):
            tf_events_files.append(tfevent)

        return tf_events_files

    def extract_data_from_event_file(
        self,
        event_file_path,
        split,
        exclude="step",
    ):
        df = pd.DataFrame()

        try:
            # Iterate through the records in one event file
            for e in tf.compat.v1.train.summary_iterator(event_file_path):
                for v in e.summary.value:
                    if v.HasField("simple_value"):
                        if split in v.tag and exclude not in v.tag:
                            df.loc[e.step, v.tag] = v.simple_value
        except Exception as e:
            print(f"{e}: {event_file_path}. Skipping file.")

        return df

    def create_dataframe(self):
        dataframes = []
        pbar = tqdm(self.log_dirs)
        for log_dir in pbar:
            pbar.set_description(
                f"Creating dataframes .../{log_dir.relative_to(self.start_path)}"  # noqa
            )
            df = pd.DataFrame()
            for version_dir in self._get_version_dirs(log_dir):
                tf_events_files = self._get_tf_events_files(version_dir)
                for tfevent in tf_events_files:
                    new_df = self.extract_data_from_event_file(
                        str(tfevent),
                        split=self.split,
                    )
                    new_df["version"] = version_dir.name
                    df = pd.concat([df, new_df])
            df.reset_index(drop=True, inplace=True)
            dataframes.append({"log_dir": log_dir, "dataframe": df})

        return dataframes
