import torch
import pandas as pd
import logging
import itertools

from .base import SearchStrategyBase
from chop.actions.search.search_space.base import SearchSpaceBase

logger = logging.getLogger(__name__)


class SearchStrategyBruteForce(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        """
        Post init setup. This is where additional config parsing and setup should be done for the subclass instance.
        """
        # self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))

    def _product_dict(self, **kwargs):
        keys = kwargs.keys()
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(keys, instance))

    def _generate_all_index_cfgs(self, cfg_index_lengths: dict[str, int]):
        for param, length in cfg_index_lengths.items():
            cfg_index_lengths[param] = list(range(length))
        return list(self._product_dict(**cfg_index_lengths))

    def search(self, search_space: SearchSpaceBase):
        """
        Perform search, and save the results.
        """
        index_lengths = search_space.choice_lengths_flattened
        index_cfgs = self._generate_all_index_cfgs(index_lengths)

        print(f"Sweeping over {len(index_cfgs)} configs...")

        all_runs = {
            "run_id": [],
            "cfg": [],
            "metrics": [],
            "scaled_metrics": [],
            "score": [],
        }

        for i, c in enumerate(index_cfgs):
            cfg = search_space.flattened_indexes_to_config(c)
            model = search_space.rebuild_model(cfg, True)

            metrics = {}

            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, cfg)
                for runner in self.sw_runner:
                    metrics |= runner(self.data_module, model, cfg)

            scaled_metrics = {}
            for metric_name in self.metric_names:
                scaled_metrics[metric_name] = (
                    self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
                )

            score = sum(scaled_metrics.values())

            print(f"Config {i}: Score {score:.5f}")

            all_runs["run_id"].append(i)
            all_runs["cfg"].append(cfg)
            all_runs["metrics"].append(metrics)
            all_runs["scaled_metrics"].append(scaled_metrics)
            all_runs["score"].append(score)

        dataframe = pd.DataFrame(all_runs)
        top = dataframe.sort_values("score", ascending=False)

        # Show top configs
        print("Top 10 Scoring Configs:")
        print(top.head(n=10))

        # Save runs
        dataframe.to_json(self.save_dir / "all.json")
        top.to_json(self.save_dir / "top.json")
